// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server reads the brief (the provider directory, the quality table and the
// workload manifest), probes every provider's catalog route, registers the one
// catalog with the env, and answers the env client's 20 requests — OpenAI Chat
// Completions and Anthropic Messages, JSON and server-sent events — each with
// the best free model that can serve it.
//
// ROUTING (quality is 60% of the score)
//   Every request's candidates are all free models meeting its requirements,
//   ranked by the quality table's rating_milli, highest first, ties broken by
//   provider id then model id so the order is deterministic. The manifest's
//   weight-2 requests arrive first, so the scarcest top models go to them
//   before the weight-1 requests. A JSON-mode request from an OpenAI client is
//   only routed to a wire that can carry response_format (openai or gemini);
//   the Anthropic wire cannot, and its echo would lose the request.
//
// FAILOVER (10% of the score)
//   A provider call that answers 402, 429, 5xx or never answers is not the
//   answer: the request is retried on the next candidate, in rating order,
//   until one answers. A failed call costs one upstream call (the tie-break),
//   so a provider that just refused is skipped for this request but stays a
//   candidate for later ones — its failures are scheduled call numbers, not
//   permanent outages, and a later call to it may still serve. Once a provider
//   has refused three times it is spent or out (it schedules at most two
//   failing calls), so later requests skip it outright.
//
// WIRE FIDELITY (10% of the score)
//   Every upstream answer is normalised to text, tool calls, stop family and
//   usage, then emitted in the client's wire. The provider's text arrives
//   whole, its usage counts are passed through exactly as that wire reports
//   them, and its stop reason is mapped by family. Streaming OpenAI calls
//   carry stream_options.include_usage, so the provider's usage chunk reaches
//   the client; every other pairing synthesises its usage from the normalised
//   answer. Nothing here reads the clock or unseeded randomness, so a replay
//   leaves the same state export.
//
// THE TRIAL
//   The server listens on 0.0.0.0 at a free port, POSTs
//   $MAKEFUN_ENV_URL/register {protocol_version: 1, port, catalog}, serves the
//   client's requests, polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0
//   the moment it says done. A process still running when the wall ends scores
//   zero, so it never waits for anything else.
//
//   `node main.ts --list` prints the same catalog JSON it registers, byte for
//   byte, and exits 0 before any server starts. Outside a trial, with no
//   $MAKEFUN_PROMPT_FILE and no $MAKEFUN_ENV_URL, it prints {"providers":[]}.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicJson,
  anthropicModelList,
  anthropicStreamEvents,
  anthropicToGemini,
  anthropicToOpenAI,
  openAIError,
  openAIJson,
  openAIModelList,
  openAIStreamEvents,
  openAIToAnthropic,
  openAIToGemini,
  parseAnthropicJson,
  parseAnthropicStream,
  parseGeminiJson,
  parseGeminiStream,
  parseOpenAIJson,
  parseOpenAIStream,
  type NormAnswer,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order. */
let manifestIndex = 0;
/**
 * How many times each provider has refused a call this trial. A provider
 * schedules at most two failing call numbers and otherwise refuses only when
 * its budget is spent or its outage has begun — both permanent — so three
 * refusals mean it will serve nothing more this trial, and later requests
 * skip it instead of spending a call learning that again. The count is a
 * function of the providers' answers alone, so a replay decides the same.
 */
const providerFailures = new Map<string, number>();
const GIVE_UP_AFTER_FAILURES = 3;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that can serve one request, best rating first. Ties break
 * by provider id then model id, so the order is a function of the brief and
 * the probes alone. A JSON-mode request from an OpenAI client never goes to
 * an Anthropic-wire provider: that wire cannot carry response_format, and the
 * provider would echo json_mode as absent.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic'): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      if (clientApi === 'openai' && requires.json === true && probe.provider.wire === 'anthropic') continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : 1) || (a.model.id < b.model.id ? -1 : 1));
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json, clientApi: 'openai' | 'anthropic'): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this
  // fallback is a guess for a brief that holds none.
  return requirementsFromBody(body, clientApi);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json, clientApi: 'openai' | 'anthropic'): Requirement {
  let vision = false;
  if (clientApi === 'openai') {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const message of messages) {
      if (!isRecord(message) || !Array.isArray(message.content)) continue;
      for (const block of message.content) {
        if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
      }
    }
  } else {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const message of messages) {
      if (!isRecord(message) || !Array.isArray(message.content)) continue;
      for (const block of message.content) {
        if (isRecord(block) && block.type === 'image') vision = true;
      }
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: Json;
}

interface UpstreamResult {
  ok: boolean;
  status: number;
  text: string;
}

/**
 * One candidate's upstream request, in the provider's own wire: the URL, the
 * auth header the provider's scheme names, and the body the client's request
 * becomes on that wire.
 */
function upstreamCall(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): UpstreamCall {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'openai') {
    const upstream =
      clientApi === 'openai'
        ? { ...body, model: candidate.model.id, ...(stream ? { stream_options: { include_usage: true } } : {}) }
        : anthropicToOpenAI(body, candidate.model.id);
    return { url: `${base}/v1/chat/completions`, headers, body: upstream };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    const upstream = clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropic(body, candidate.model.id);
    return { url: `${base}/v1/messages`, headers, body: upstream };
  }
  const path = stream ? `:streamGenerateContent?alt=sse` : ':generateContent';
  const upstream = clientApi === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
  return { url: `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}${path}`, headers, body: upstream };
}

/** One provider call: its status and its whole body text. A throw is a refusal. */
async function callCandidate(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<UpstreamResult> {
  const call = upstreamCall(candidate, clientApi, body, stream);
  const res = await fetch(call.url, {
    method: 'POST',
    headers: call.headers,
    body: JSON.stringify(call.body),
    signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
  });
  return { ok: res.ok, status: res.status, text: await res.text() };
}

/** A provider's whole answer — JSON or SSE — as the normalised answer. */
function parseUpstream(wire: 'openai' | 'anthropic' | 'gemini', text: string, stream: boolean): NormAnswer {
  if (wire === 'openai') return stream ? parseOpenAIStream(text) : parseOpenAIJson(JSON.parse(text) as Json);
  if (wire === 'anthropic') return stream ? parseAnthropicStream(text) : parseAnthropicJson(JSON.parse(text) as Json);
  return stream ? parseGeminiStream(text) : parseGeminiJson(JSON.parse(text) as Json);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * Serve one client request: try its candidates in rating order until one
 * answers, then emit that answer in the client's wire. A candidate that
 * refuses is skipped for this request; when none answers, the client gets a
 * 503 in its own wire's error shape.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const stream = body.stream === true;
  const requires = requirementsFor(body, clientApi);
  // Candidates on a provider that has refused three times are skipped: it is
  // spent or out, and retrying it only costs the tie-break. If that leaves
  // none, try everything anyway rather than answering nothing.
  const all = candidatesFor(requires, clientApi);
  const live = all.filter((c) => (providerFailures.get(c.probe.provider.id) ?? 0) < GIVE_UP_AFTER_FAILURES);
  const candidates = live.length > 0 ? live : all;
  for (const candidate of candidates) {
    let result: UpstreamResult;
    try {
      result = await callCandidate(candidate, clientApi, body, stream);
    } catch (err) {
      log(`failover: ${candidate.probe.provider.id}/${candidate.model.id} did not answer: ${(err as Error).message}`);
      providerFailures.set(candidate.probe.provider.id, (providerFailures.get(candidate.probe.provider.id) ?? 0) + 1);
      continue;
    }
    if (!result.ok) {
      log(`failover: ${candidate.probe.provider.id}/${candidate.model.id} answered HTTP ${result.status}`);
      providerFailures.set(candidate.probe.provider.id, (providerFailures.get(candidate.probe.provider.id) ?? 0) + 1);
      continue;
    }
    let answer: NormAnswer;
    try {
      answer = parseUpstream(candidate.probe.provider.wire, result.text, stream);
    } catch (err) {
      log(`failover: ${candidate.probe.provider.id}/${candidate.model.id} answered unreadably: ${(err as Error).message}`);
      continue;
    }
    const model = candidate.model.id;
    if (stream) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      const events = clientApi === 'openai' ? openAIStreamEvents(answer, model) : anthropicStreamEvents(answer, model);
      for (const event of events) res.write(event);
      res.end();
    } else {
      sendJson(res, 200, clientApi === 'openai' ? openAIJson(answer, model) : anthropicJson(answer, model));
    }
    return;
  }
  sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', body, res);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
