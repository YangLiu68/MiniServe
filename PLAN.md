# MiniServe — Plan

MiniServe is an educational but real text-generation server. The portfolio point is
**measured** scheduler/latency engineering with honest comparison to mature engines
(vLLM, TGI, llama.cpp) — not a claim to beat them.

## Environment inventory (inspected 2026-10-14)

| Item | Finding |
|---|---|
| Hardware | Apple M4, 10 cores, 16 GB unified memory, macOS 15.3.2. **No CUDA GPU.** |
| Accelerator | PyTorch MPS available; CPU fallback. CUDA path is written but cannot be tested locally. |
| Python | 3.12.10 via uv-managed `.venv`; uv 0.9.2 |
| Key deps | torch 2.14.1, transformers 5.19.0, fastapi 0.143.1, starlette 1.8.0, uvicorn 0.54.0, pydantic 2.14.1 |
| Docker | Not installed → Dockerfile (M7) can be written but not built/run locally. |
| Credentials | None present or required (no HF token; default model is ungated). |
| Default model | `HuggingFaceTB/SmolLM2-135M-Instruct`, Apache-2.0, ungated, pinned revision `12fd25f77366fa6b3b4b768ec3050bf629380bac` |

## Design principles

- One runnable vertical slice first; grow the engine in place (queue → scheduler → batcher).
- Every backend implements the same small interface; a **deterministic toy backend** (no torch)
  drives scheduling/streaming tests, and a **tiny randomly-initialised GPT-2** (no download)
  exercises the real PyTorch/Transformers code path in tests.
- OpenAI-compatible `/v1/completions` subset, so the M6 load generator can hit MiniServe and
  vLLM with the same client (apples-to-apples harness).
- Nothing is claimed as measured unless a script actually ran; results are recorded with
  hardware, versions, and command lines.

## Milestones

### M1 — Single-request generation endpoint ✅ (see progress log)
Scope: model/tokenizer lifecycle, validation, token caps, cancellation, timeouts, SSE streaming,
deterministic toy-model tests, basic safety (auth, loopback default, rate + token limits).

Acceptance criteria:
1. `POST /v1/completions` returns an OpenAI-style completion (non-stream) and SSE chunks
   ending in `data: [DONE]` (stream). Streamed deltas concatenate to the non-stream text.
2. Model loads in the background; `/readyz` is 503 until loaded+warmed, then 200;
   `/healthz` is liveness. Completions return 503 while loading.
3. Validation: empty prompt, unknown fields, bad temperature/top_p → 422; `max_tokens` above the
   server cap, or prompt tokens above the prompt cap / context window → 400; wrong `model` → 404.
4. `max_tokens` enforced exactly (`finish_reason="length"`); EOS → `finish_reason="stop"`.
5. Cancellation: client disconnect (real socket close against a live uvicorn server) and
   `POST /v1/requests/{id}/cancel` both stop generation within one decode step.
6. Timeouts: per-request deadline (server max, client may only lower) covers queue wait +
   generation; streaming reports `finish_reason="timeout"` in-band, non-stream returns 504.
7. Single model instance is never run concurrently; extra requests wait in a bounded FIFO
   (`MAX_PENDING`), overflow → 503 + `Retry-After`. (Fair scheduling is M2.)
8. Safety: refuses to bind a non-loopback host without `MINISERVE_API_KEY` unless explicitly
   overridden; bearer auth when a key is set; per-client token-bucket rate limit → 429.
9. Toy-model tests are deterministic; tiny-GPT-2 tests show our manual decode loop matches
   `model.generate()` greedy output, and seeded sampling is reproducible.
10. Real-model smoke run (SmolLM2-135M on MPS and CPU) executes end-to-end via HTTP.

### M2 — Concurrent queue, fairness, admission control, backpressure
- Replace FIFO with a scheduler queue: per-client fair share (round-robin / deficit), priority by
  deadline optional; admission control on queued prompt tokens + estimated KV memory, not just
  request count; bounded per-request event buffers; Prometheus metrics (`/metrics`).
- Acceptance: unit tests for fairness ordering, admission rejections, cancellation of queued and
  running requests; an experiment quantifying head-of-line blocking (short request TTFT behind a
  long request) with toy backend (controlled delays) and real model.

### M3 — Dynamic (continuous) batching scheduler
- Iteration-level batching: prefill new requests and decode active ones in shared forward passes
  with padding/attention masks; documented trade-offs (max batch size, prefill/decode
  interleaving, wait window, padding waste).
- Acceptance: batched output equals single-request output (greedy) on tiny model; throughput vs
  latency curves vs the sequential M1/M2 engine at matched model/hardware/dtype.

### M4 — KV-cache reuse
- Instrument Transformers `DynamicCache` (integration, clearly labelled as such) and implement an
  original component: prefix cache keyed by token-prefix hash with LRU eviction and a byte budget.
- Acceptance: cache-hit outputs identical to cold outputs; memory accounting tests; eviction tests;
  measured TTFT reduction for shared-prefix workloads.

### M5 — Optional quantized backend or external vLLM adapter
- Most likely: an adapter that points the M6 load generator at an external OpenAI-compatible server
  (vLLM / llama.cpp server), plus a compatibility matrix and caveats (hardware, dtype, kernels,
  sampling defaults, tokenizer). vLLM needs CUDA → cannot run on this Mac; will be labelled
  "not executed" unless a GPU host is provided.

### M6 — Load generator + benchmark report
- Open-loop Poisson arrivals and closed-loop concurrency sweeps; prompt/output length
  distributions; P50/P95/P99 TTFT, inter-token latency, tokens/s, throughput, errors, process
  RSS / MPS / CUDA memory. Raw results as JSONL + generated BENCHMARK.md tables.

### M7 — Ops hardening
- Dockerfile, readiness/liveness (started in M1), structured JSON logs, metrics, graceful drain on
  SIGTERM, GitHub Actions (lint + toy/tiny-model tests on CPU, no secrets), architecture diagram,
  demo walkthrough.

## Explicit non-goals
- Beating vLLM/TGI throughput, custom CUDA/Metal kernels, paged attention, tensor parallelism.
- Chat templates / `/v1/chat/completions`, stop strings, logprobs, `n>1`, beam search (may revisit).
- Multi-tenant auth, TLS termination, public Internet exposure (put behind a real gateway).
- Multi-process / multi-GPU serving; persistence of cache across restarts.
- Training or fine-tuning anything.

## Progress log

### 2026-10-14 — M1 started
- Inspected environment (table above). Created standalone git repo (the parent home directory is
  itself a git repo; MiniServe now has its own `.git`).

### 2026-10-14 — M1 complete
- **Implemented:** `config.py`, `schemas.py`, `server.py`, `engine.py`, `detokenizer.py`,
  `ratelimit.py`, `backends/{base,toy,hf}.py`, `__main__.py`; tests in `tests/`; docs README,
  ARCHITECTURE, TESTING, BENCHMARK, LICENSE_NOTES, docs/DEMO; `.env.example`, `.gitignore`.
- **Tests:** `uv run pytest` → 55 passed, 3 skipped (opt-in real-model); real-model opt-in → 3
  passed; ruff clean. Mutation checks confirm the disconnect and post-step-cancel tests fail when
  the corresponding code is removed. All 10 acceptance criteria mapped to evidence in TESTING.md.
- **Real-model end-to-end** (SmolLM2-135M, MPS fp16, via `miniserve` + curl): bind refusal, readiness
  transition, auth, JSON, SSE, disconnect cancellation, SIGTERM shutdown — see docs/DEMO.md.
- **Bugs found and fixed:** (1) app closed injected backends at shutdown; (2) a disconnect during
  an uninterruptible step whose token was EOS was reported as `stop` instead of `cancelled`.
  Both have regression tests.
- **Findings:** greedy output identical on CPU fp32 / MPS fp16 / MPS fp32. Host was saturated by
  unrelated processes (load avg 93–121), so **no performance numbers are valid yet**.
- **Remaining risks:** forward passes are not interruptible (long-prefill cancel latency);
  FIFO head-of-line blocking (intended for M1, to be measured in M2); timing-based tests could
  flake on starved CI runners; CUDA path untested; MPS fp16 vs fp32 default is unmeasured;
  `resource_tracker` leaked-semaphore warning at exit not investigated; project license TBD.
- **Next step (M2):** fair scheduler queue (per-client), token/memory-aware admission, `/metrics`,
  and a head-of-line-blocking experiment (short-request TTFT behind long requests) on the toy
  backend with controlled delays, then on the real model once the host is quiet.
