// lib/wire.ts — the client wires and the three upstream wires.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions)
// or an Anthropic one (POST /v1/messages), with a JSON body or with server-sent
// events. The entry answers from a provider on the openai, anthropic or gemini
// wire, so every pairing needs a request builder and an answer translator:
//
//   client openai    -> upstream openai:    pass the body through (model set,
//                      stream_options.include_usage added for a stream, so the
//                      provider's usage chunk reaches the client)
//   client openai    -> upstream anthropic: rewrite to a Messages request
//   client openai    -> upstream gemini:    rewrite to generateContent
//   client anthropic -> upstream anthropic: pass the body through (model set)
//   client anthropic -> upstream openai:    rewrite to Chat Completions
//   client anthropic -> upstream gemini:    rewrite to generateContent
//
// Every translator preserves the fields the providers echo back and the client
// compares with what it sent: the system prompt, the user text (images included),
// the tool names, the stop sequences, max_tokens and JSON mode. JSON mode travels
// as response_format on the OpenAI wire and as responseMimeType on the Gemini
// wire; the Anthropic wire cannot carry it, so a JSON-mode OpenAI request is
// never routed to an Anthropic-wire provider.
//
// Answers are normalised to one shape — text, tool calls, stop family, usage —
// and then emitted in the client's wire. Usage counts are the provider's own,
// passed through exactly: the client checks them against what the provider
// reported. Stop reasons map by family (stop/end_turn/STOP, length/max_tokens,
// tool_calls/tool_use, content_filter/refusal).
//
// Everything here is a pure function of its arguments: no clock and no
// randomness, so the same body and the same provider answer give the same bytes
// on a replay. The two model listings use fixed timestamps for the same reason.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** The id every OpenAI-shaped answer carries (the providers use one id too). */
const OPENAI_ANSWER_ID = 'chatcmpl-sim';
/** The id every Anthropic-shaped answer carries. */
const ANTHROPIC_ANSWER_ID = 'msg_sim';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** One server-sent event: `event: <name>` when named, then the data line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One parsed SSE frame: its event name (null when unnamed) and its data lines. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Split SSE text into frames. */
export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** Split text the way the providers split their deltas: 13 characters a piece. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** One tool call in a provider's answer. */
export interface ToolCall {
  id: string;
  name: string;
  /** The arguments as the provider wrote them, a JSON string. */
  arguments: string;
}

/** A provider's stop reason, in the four families the wires share. */
export type StopFamily = 'stop' | 'length' | 'tool_calls' | 'content_filter';

/** A provider's answer, normalised across the three wires. */
export interface NormAnswer {
  text: string;
  toolCalls: ToolCall[];
  stop: StopFamily;
  usage: { input: number; output: number };
}

function emptyAnswer(): NormAnswer {
  return { text: '', toolCalls: [], stop: 'stop', usage: { input: 0, output: 0 } };
}

function openAIFinishToStop(finish: unknown): StopFamily {
  if (finish === 'length') return 'length';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_calls';
  if (finish === 'content_filter') return 'content_filter';
  return 'stop';
}

function anthropicStopToStop(stop: unknown): StopFamily {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

function geminiFinishToStop(finish: unknown): StopFamily {
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY') return 'content_filter';
  return 'stop';
}

function stopToOpenAI(stop: StopFamily): string {
  if (stop === 'length') return 'length';
  if (stop === 'tool_calls') return 'tool_calls';
  if (stop === 'content_filter') return 'content_filter';
  return 'stop';
}

function stopToAnthropic(stop: StopFamily): string {
  if (stop === 'length') return 'max_tokens';
  if (stop === 'tool_calls') return 'tool_use';
  if (stop === 'content_filter') return 'refusal';
  return 'end_turn';
}

// ---------------------------------------------------------------------------
// Parsing a provider's answer into the normalised shape.
// ---------------------------------------------------------------------------

/** A Chat Completions JSON answer as a normalised answer. */
export function parseOpenAIJson(body: Json): NormAnswer {
  const out = emptyAnswer();
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  out.text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    out.toolCalls.push({
      id: typeof call.id === 'string' ? call.id : 'call_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      arguments: typeof call.function.arguments === 'string' ? call.function.arguments : JSON.stringify(argumentsOf(call.function.arguments)),
    });
  }
  out.stop = openAIFinishToStop(choice.finish_reason);
  const usage = isRecord(body.usage) ? body.usage : {};
  out.usage = { input: numberOrZero(usage.prompt_tokens), output: numberOrZero(usage.completion_tokens) };
  return out;
}

/** A Chat Completions SSE stream as a normalised answer. */
export function parseOpenAIStream(text: string): NormAnswer {
  const out = emptyAnswer();
  const byIndex = new Map<number, ToolCall>();
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      out.usage = { input: numberOrZero(chunk.usage.prompt_tokens), output: numberOrZero(chunk.usage.completion_tokens) };
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out.text += delta.content;
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const index = typeof call.index === 'number' ? call.index : byIndex.size;
      let entry = byIndex.get(index);
      if (entry === undefined) {
        const fn = isRecord(call.function) ? call.function : {};
        entry = {
          id: typeof call.id === 'string' && call.id !== '' ? call.id : `call_${index}`,
          name: typeof fn.name === 'string' ? fn.name : '',
          arguments: '',
        };
        byIndex.set(index, entry);
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.name === 'string' && fn.name !== '') entry.name = fn.name;
      if (typeof call.id === 'string' && call.id !== '') entry.id = call.id;
      if (typeof fn.arguments === 'string') entry.arguments += fn.arguments;
    }
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) out.stop = openAIFinishToStop(choice.finish_reason);
  }
  out.toolCalls = [...byIndex.entries()].sort((a, b) => a[0] - b[0]).map(([, call]) => call);
  return out;
}

/** A Messages JSON answer as a normalised answer. */
export function parseAnthropicJson(body: Json): NormAnswer {
  const out = emptyAnswer();
  const content = Array.isArray(body.content) ? body.content : [];
  const parts: string[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
    else if (block.type === 'tool_use') {
      out.toolCalls.push({
        id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
        name: typeof block.name === 'string' ? block.name : '',
        arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
      });
    }
  }
  out.text = parts.join('');
  out.stop = anthropicStopToStop(body.stop_reason);
  const usage = isRecord(body.usage) ? body.usage : {};
  out.usage = { input: numberOrZero(usage.input_tokens), output: numberOrZero(usage.output_tokens) };
  return out;
}

/** A Messages SSE stream as a normalised answer. */
export function parseAnthropicStream(text: string): NormAnswer {
  const out = emptyAnswer();
  const byIndex = new Map<number, ToolCall>();
  for (const frame of parseSseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      out.usage.input = numberOrZero(data.message.usage.input_tokens);
    }
    if (frame.event === 'content_block_start' && isRecord(data.content_block) && data.content_block.type === 'tool_use') {
      const index = typeof data.index === 'number' ? data.index : byIndex.size;
      byIndex.set(index, {
        id: typeof data.content_block.id === 'string' ? data.content_block.id : `toolu_${index}`,
        name: typeof data.content_block.name === 'string' ? data.content_block.name : '',
        arguments: '',
      });
    }
    if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      const index = typeof data.index === 'number' ? data.index : -1;
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') out.text += data.delta.text;
      else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        const entry = byIndex.get(index);
        if (entry !== undefined) entry.arguments += data.delta.partial_json;
      }
    }
    if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') out.stop = anthropicStopToStop(data.delta.stop_reason);
      if (isRecord(data.usage)) out.usage.output = numberOrZero(data.usage.output_tokens);
    }
  }
  out.toolCalls = [...byIndex.entries()].sort((a, b) => a[0] - b[0]).map(([, call]) => call);
  return out;
}

/** The text of a Gemini candidate's parts, and any functionCall parts. */
function geminiParts(parts: unknown): { text: string; calls: ToolCall[] } {
  const texts: string[] = [];
  const calls: ToolCall[] = [];
  if (!Array.isArray(parts)) return { text: '', calls };
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') texts.push(part.text);
    else if (isRecord(part.functionCall)) {
      calls.push({
        id: typeof part.functionCall.name === 'string' ? part.functionCall.name : 'call_unknown',
        name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
        arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
      });
    }
  }
  return { text: texts.join(''), calls };
}

/** A generateContent JSON answer as a normalised answer. */
export function parseGeminiJson(body: Json): NormAnswer {
  const out = emptyAnswer();
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parsed = geminiParts(content.parts);
  out.text = parsed.text;
  out.toolCalls = parsed.calls;
  out.stop = geminiFinishToStop(first.finishReason);
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  out.usage = { input: numberOrZero(usage.promptTokenCount), output: numberOrZero(usage.candidatesTokenCount) };
  return out;
}

/** A streamGenerateContent SSE stream as a normalised answer. */
export function parseGeminiStream(text: string): NormAnswer {
  const out = emptyAnswer();
  for (const frame of parseSseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(first.content) ? first.content : {};
    const parsed = geminiParts(content.parts);
    out.text += parsed.text;
    out.toolCalls.push(...parsed.calls);
    if (typeof first.finishReason === 'string') out.stop = geminiFinishToStop(first.finishReason);
    if (isRecord(data.usageMetadata)) {
      out.usage = { input: numberOrZero(data.usageMetadata.promptTokenCount), output: numberOrZero(data.usageMetadata.candidatesTokenCount) };
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Emitting a normalised answer in the client's wire.
// ---------------------------------------------------------------------------

/** A normalised answer as a Chat Completions JSON object. */
export function openAIJson(answer: NormAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((c) => ({
      id: c.id,
      type: 'function',
      function: { name: c.name, arguments: c.arguments },
    }));
  }
  return {
    id: OPENAI_ANSWER_ID,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: stopToOpenAI(answer.stop), logprobs: null }],
    usage: {
      prompt_tokens: answer.usage.input,
      completion_tokens: answer.usage.output,
      total_tokens: answer.usage.input + answer.usage.output,
    },
  };
}

/** A normalised answer as Chat Completions SSE events, ending with [DONE]. */
export function openAIStreamEvents(answer: NormAnswer, model: string): string[] {
  const events: string[] = [];
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: OPENAI_ANSWER_ID,
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  events.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  for (const piece of splitText(answer.text)) events.push(sseEvent(null, chunk({ content: piece }, null)));
  answer.toolCalls.forEach((call, index) => {
    events.push(
      sseEvent(
        null,
        chunk({ tool_calls: [{ index, id: call.id, type: 'function', function: { name: call.name, arguments: '' } }] }, null),
      ),
    );
    if (call.arguments !== '') {
      events.push(sseEvent(null, chunk({ tool_calls: [{ index, function: { arguments: call.arguments } }] }, null)));
    }
  });
  events.push(sseEvent(null, chunk({}, stopToOpenAI(answer.stop))));
  events.push(
    sseEvent(null, {
      id: OPENAI_ANSWER_ID,
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.usage.input,
        completion_tokens: answer.usage.output,
        total_tokens: answer.usage.input + answer.usage.output,
      },
    }),
  );
  events.push('data: [DONE]\n\n');
  return events;
}

/** A normalised answer as a Messages JSON object. */
export function anthropicJson(answer: NormAnswer, model: string): Json {
  const content: Json[] = [];
  if (answer.text !== '' || answer.toolCalls.length === 0) content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({ type: 'tool_use', id: call.id, name: call.name, input: argumentsOf(call.arguments) });
  }
  return {
    id: ANTHROPIC_ANSWER_ID,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopToAnthropic(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.usage.input, output_tokens: answer.usage.output },
  };
}

/** A normalised answer as the Messages SSE event order. */
export function anthropicStreamEvents(answer: NormAnswer, model: string): string[] {
  const events: string[] = [];
  events.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: ANTHROPIC_ANSWER_ID,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.usage.input, output_tokens: 0 },
      },
    }),
  );
  let index = 0;
  if (answer.text !== '' || answer.toolCalls.length === 0) {
    const textIndex = index;
    index += 1;
    events.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
    for (const piece of splitText(answer.text)) {
      events.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: piece } }));
    }
    events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
  }
  for (const call of answer.toolCalls) {
    const toolIndex = index;
    index += 1;
    events.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index: toolIndex,
        content_block: { type: 'tool_use', id: call.id, name: call.name, input: {} },
      }),
    );
    if (call.arguments !== '') {
      events.push(
        sseEvent('content_block_delta', { type: 'content_block_delta', index: toolIndex, delta: { type: 'input_json_delta', partial_json: call.arguments } }),
      );
    }
    events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: toolIndex }));
  }
  events.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopToAnthropic(answer.stop), stop_sequence: null },
      usage: { output_tokens: answer.usage.output },
    }),
  );
  events.push(sseEvent('message_stop', { type: 'message_stop' }));
  return events;
}

// ---------------------------------------------------------------------------
// Building an upstream request in the provider's wire.
// ---------------------------------------------------------------------------

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** One Anthropic content block as an OpenAI content part; null for the rest. */
function anthropicBlockToPart(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming
 * `model`: the system prompt becomes a system message, image blocks become
 * image_url data URLs, tool_use blocks become tool_calls, tool_result blocks
 * become tool messages, and tools, tool_choice, the output cap, temperature,
 * top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
        });
      }
      const text = textOf(content);
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({ role: 'tool', tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', content: toolResultText(block.content) });
        continue;
      }
      const part = anthropicBlockToPart(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One OpenAI content part as an Anthropic content block; null for the rest. */
function openAIPartToBlock(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) {
      return { type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/** The stop sequences of an OpenAI request, as a string array. */
function openAIStops(body: Json): string[] {
  if (typeof body.stop === 'string') return [body.stop];
  if (Array.isArray(body.stop)) return body.stop.filter((s): s is string => typeof s === 'string');
  return [];
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming
 * `model`. JSON mode cannot travel on this wire: the caller never routes a
 * JSON-mode request here.
 */
export function openAIToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const content: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        content.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content });
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: toolResultText(raw.content) }],
      });
      continue;
    }
    if (role !== 'user') continue;
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    const blocks: Json[] = [];
    for (const part of raw.content) {
      if (!isRecord(part)) continue;
      const block = openAIPartToBlock(part);
      if (block !== null) blocks.push(block);
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = openAIStops(body);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** An OpenAI image_url part as a Gemini part; null when it is not an image. */
function openAIImageToGeminiPart(part: Json): Json | null {
  if (part.type !== 'image_url' || !isRecord(part.image_url) || typeof part.image_url.url !== 'string') return null;
  const url = part.image_url.url;
  const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (data !== null) return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
  return { fileData: { mimeType: 'image/png', fileUri: url } };
}

/** An Anthropic image block as a Gemini part; null when it is not an image. */
function anthropicImageToGeminiPart(block: Json): Json | null {
  if (block.type !== 'image' || !isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { fileData: { mimeType: 'image/png', fileUri: source.url } };
  return null;
}

function geminiGenerationConfig(maxTokens: number | undefined, temperature: unknown, topP: unknown, stops: string[], jsonMode: boolean): Json {
  const config: Json = {};
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (typeof temperature === 'number') config.temperature = temperature;
  if (typeof topP === 'number') config.topP = topP;
  if (stops.length > 0) config.stopSequences = stops;
  if (jsonMode) config.responseMimeType = 'application/json';
  return config;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body. The
 * model travels in the URL, not the body.
 */
export function openAIToGemini(body: Json): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      contents.push({ role: 'user', parts: [{ text: toolResultText(raw.content) }] });
      continue;
    }
    if (raw.role !== 'user') continue;
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
        else {
          const image = openAIImageToGeminiPart(part);
          if (image !== null) parts.push(image);
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }

  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const config = geminiGenerationConfig(maxTokens, body.temperature, body.top_p, openAIStops(body), format === 'json_object' || format === 'json_schema');
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent body. The
 * Anthropic wire never carries JSON mode, so no responseMimeType is set.
 */
export function anthropicToGemini(body: Json): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      if (typeof content === 'string') {
        if (content !== '') parts.push({ text: content });
      } else if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'tool_use') {
            parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
          }
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role !== 'user') continue;
    const parts: Json[] = [];
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'tool_result') {
          parts.push({ functionResponse: { name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', response: { content: toolResultText(block.content) } } });
        } else {
          const image = anthropicImageToGeminiPart(block);
          if (image !== null) parts.push(image);
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }

  const out: Json = { contents };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const stops = Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === 'string') : [];
  const config = geminiGenerationConfig(typeof body.max_tokens === 'number' ? body.max_tokens : undefined, body.temperature, body.top_p, stops, false);
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}
