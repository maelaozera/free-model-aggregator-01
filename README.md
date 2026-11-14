# Free Model Aggregator

One server that serves the OpenAI Chat Completions API and the Anthropic
Messages API over ten simulated free-tier model providers, in JSON and in
server-sent events. Entry for make.fun `free-model-aggregator-01`.

## Run

```sh
node main.ts            # serve: reads $MAKEFUN_PROMPT_FILE, registers at $MAKEFUN_ENV_URL
node main.ts --list     # print the catalog document and exit 0
```

Node 22, standard library only — nothing is installed.

## How it works

- **Catalog**: at startup the server probes every provider's catalog route in
  the provider's own wire (OpenAI, Anthropic or Gemini) and builds one
  catalog document: provider id, base URL, key env var, wire, and each model
  with its free flag and context length. `--list` prints that same document.
- **Routing**: every request's candidates are all free models meeting its
  requirements (tools, vision, JSON mode, minimum context), ranked by the
  quality table's `rating_milli`, highest first. A JSON-mode request from an
  OpenAI client is only routed to a wire that can carry `response_format`
  (OpenAI or Gemini). The manifest's weight-2 requests arrive first, so the
  scarcest top models serve them before the weight-1 requests.
- **Failover**: a provider call that answers 402, 429, 5xx or never answers is
  retried on the next candidate in rating order until one answers.
- **Wires**: every pairing of client wire (OpenAI, Anthropic) and upstream
  wire (OpenAI, Anthropic, Gemini) is translated both ways, preserving the
  system prompt, user text and images, tool definitions, stop sequences,
  `max_tokens` and JSON mode. Answers are normalised to text, tool calls,
  stop family and usage, then emitted in the client's wire with the
  provider's usage counts passed through exactly. Streaming OpenAI calls
  carry `stream_options.include_usage`.
- **Trial**: the server registers its port and catalog at
  `POST $MAKEFUN_ENV_URL/register`, serves the client's requests, polls
  `GET $MAKEFUN_ENV_URL/work` every 100 ms and exits 0 when the workload is
  done. Nothing reads the clock or unseeded randomness, so a replay leaves
  the same state export.

## Files

- `main.ts` — the server: brief, catalog, routing, failover, register, poll
- `lib/providers.ts` — brief parsing, catalog probe, quality table
- `lib/wire.ts` — the six client/upstream wire translations, JSON and SSE
- `LICENSE` — MIT
- `.makefun/owner.json` — the claim file naming the entrant's wallet
