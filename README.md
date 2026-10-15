# MiniServe

An educational but real LLM text-generation server, built to make **request scheduling,
streaming, caching and latency measurement** visible and testable. It is not a competitor to
vLLM / TGI / llama.cpp, and does not claim to be faster than them; later milestones will compare
against them with documented caveats.

**Status: Milestone 1 of 7 complete** — single-request generation with lifecycle, validation,
token caps, cancellation, timeouts and SSE streaming. See [PLAN.md](PLAN.md) for the roadmap and
progress log.

| Doc | What's in it |
|---|---|
| [PLAN.md](PLAN.md) | Milestones, acceptance criteria, non-goals, progress log |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Request flow, threading model, cancellation/timeout paths, trade-offs |
| [TESTING.md](TESTING.md) | How to run tests, what each test proves, latest results |
| [BENCHMARK.md](BENCHMARK.md) | Benchmark status and methodology (no baselines yet — see why) |
| [LICENSE_NOTES.md](LICENSE_NOTES.md) | Model and dependency licenses |
| [docs/DEMO.md](docs/DEMO.md) | Demo walkthrough with real captured output |

## Quickstart

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra hf                # server + torch/transformers + dev tools
uv run pytest                     # toy + tiny-model tests, no downloads (~1-2 min)

# Real model (downloads ~270 MB SmolLM2-135M-Instruct on first start):
MINISERVE_API_KEY=demo-key uv run miniserve
# No-download alternatives:
MINISERVE_BACKEND=toy  uv run miniserve    # deterministic echo model, no torch needed
MINISERVE_BACKEND=tiny uv run miniserve    # random 2-layer GPT-2 (gibberish, real torch path)
```

Wait for readiness, then generate:

```bash
curl -s localhost:8000/readyz        # 503 {"status":"loading"} until model is loaded + warmed up

curl -s localhost:8000/v1/completions -H 'Authorization: Bearer demo-key' \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The three primary colors are","max_tokens":24,"temperature":0}'

curl -sN localhost:8000/v1/completions -H 'Authorization: Bearer demo-key' \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"List three fruits:","max_tokens":12,"temperature":0,"stream":true}'
```

## API (M1)

OpenAI Completions-compatible subset, so one load generator can later target MiniServe and vLLM.

| Endpoint | Auth | Purpose |
|---|---|---|
| `POST /v1/completions` | yes | `prompt`, `max_tokens`, `temperature`, `top_p`, `seed`, `stream`, `model`, and extension `timeout_s` |
| `POST /v1/requests/{id}/cancel` | yes | Cancel a queued or running request (id from `X-Request-ID` header) |
| `GET /v1/models` | yes | Served model id |
| `GET /healthz` | no | Liveness (503 only if model loading failed) |
| `GET /readyz` | no | Readiness (200 once loaded + warmed up) |

- **finish_reason**: `stop` (EOS), `length` (`max_tokens`), plus MiniServe-specific `cancelled`
  and `timeout`.
- **Streaming**: `text/event-stream`, one `data: {json}` chunk per stable text delta, a final chunk
  carrying `finish_reason`, `usage` and server timings, then `data: [DONE]`. Errors after the
  stream has started are reported in-band (HTTP status is already 200).
- **Non-stream timeouts** return 504 with the partial completion under `partial`.
- **Unknown request fields are rejected** (422), e.g. `n`, `logprobs`, `stop` — not silently
  ignored.
- Every response includes `miniserve.timings`: `queue_ms`, `ttft_ms`, `total_ms`,
  `decode_tokens_per_s` (server-side, measured from request arrival).

## Safety defaults

- Binds `127.0.0.1` by default; **refuses to start on a non-loopback host without
  `MINISERVE_API_KEY`** (override only with `MINISERVE_ALLOW_UNAUTHENTICATED_PUBLIC=1` behind a
  trusted gateway).
- Bearer-token auth when a key is set (constant-time comparison); probes stay unauthenticated.
- Per-client-IP token-bucket rate limit (429 + `Retry-After`), prompt character/token caps,
  server-wide `max_tokens` cap, context-window check, per-request deadline, bounded wait queue
  (503 + `Retry-After` on overflow).
- Still not meant for direct Internet exposure: no TLS, single shared key, in-memory limits.

All settings are environment variables; see [.env.example](.env.example).

## Project layout

```
src/miniserve/
  config.py        Settings (MINISERVE_* env vars), bind-safety check
  schemas.py       Request model (OpenAI subset + timeout_s)
  server.py        FastAPI app: lifecycle, auth, limits, JSON + SSE responses
  engine.py        Request handles, FIFO, worker thread, cancellation, deadlines
  detokenizer.py   Incremental (streaming-safe) detokenization
  ratelimit.py     Token-bucket limiter with bounded memory
  backends/        toy (echo, no torch), hf (Transformers + manual KV-cache decode loop)
tests/             toy, tiny-random-GPT-2, live-uvicorn, opt-in real-model tests
```
