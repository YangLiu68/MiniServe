# MiniServe — Plan

MiniServe is an educational but real text-generation server. The portfolio point is
**measured** scheduler/latency engineering with honest comparison to mature engines
(vLLM, TGI, llama.cpp) — not a claim to beat them.

## Environment inventory (inspected 2026-10-14)

| Item | Finding |
|---|---|
| Hardware | Apple M4, 10 cores, 16 GB unified memory, macOS 15.3.2. **No CUDA GPU.** |
| Accelerator | PyTorch MPS; CPU fallback. CUDA paths written but untestable here. |
| Shared machine | Other projects run heavy jobs concurrently (load average 10–120 observed). A local resource manager (`rq`) leases cores/RAM/GPU; every heavy job here ran under a lease and its status is recorded with benchmark results. |
| Python / tools | 3.12.10, uv 0.9.2; torch 2.14.1, transformers 5.19.0, fastapi 0.143.1, uvicorn 0.54.0 |
| Docker | Not installed → Dockerfile/compose written; CI builds and smoke-tests the image (not yet run: repo not pushed). |
| Credentials | None present or required (default model is ungated). |
| Default model | `HuggingFaceTB/SmolLM2-135M-Instruct`, Apache-2.0, pinned revision `12fd25f7…` |
| External engines | Ollama 0.14.2 installed (llama.cpp + Metal) → usable comparison target. vLLM needs CUDA → template only. |

## Design principles

- One runnable vertical slice first; grow the engine in place (queue → scheduler → batcher →
  paged cache) without breaking the request/handle interface.
- Everything that decides *what runs when* is plain Python, testable with a fake or toy model;
  numerics are pinned against Transformers.
- OpenAI-compatible wire format so the same load generator drives MiniServe, Ollama and vLLM.
- Nothing is claimed as measured unless a script ran; results carry hardware, versions,
  commit, load average and the resource manager's status.

## Milestones

| | Milestone | Status | Evidence |
|---|---|---|---|
| M1 | Single-request endpoint: lifecycle, validation, caps, cancellation, timeouts, SSE, toy tests | ✅ | TESTING.md; docs/DEMO.md |
| M2 | Concurrent queue: fairness, admission control, backpressure, bounded memory, per-request cancellation; measure head-of-line blocking | ✅ | `policy.py` (FCFS, VTC), engine admission (in-flight, per tenant, queued tokens, KV capacity); BENCHMARK.md §HOL, §Fairness |
| M3 | Dynamic batching with documented trade-offs; throughput vs latency vs sequential baseline at matched settings | ✅ | `scheduler.py` (continuous batching, decode-first chunked prefill, preemption); BENCHMARK.md §Batching |
| M4 | KV-cache reuse; integration vs original clearly distinguished; correctness + memory tests | ✅ | `hf` backend = Transformers `DynamicCache` (integration); `block_manager.py` + `native.py` = original paged cache + prefix caching; 10 block-manager tests, prefix/preemption exactness tests; BENCHMARK.md §Prefix caching |
| M5 | Quantized backend or external comparison adapter; compatibility + caveats | ✅ | int8 W8A16 + quality eval; Ollama comparison suite (run) and vLLM template (not run); model compatibility matrix |
| M6 | Load generator + benchmark report (arrival rates, lengths, concurrency, P50/P95 TTFT, tok/s, memory, errors, tails) | ✅ | `miniserve-bench`; BENCHMARK.md |
| M7 | Docker, health/readiness, structured logs, metrics, graceful shutdown, CI, architecture diagram, demo | ✅ (CI/Docker not executed locally) | Dockerfile, compose, `.github/workflows/ci.yml`, `/metrics`, JSON logs, SIGTERM drain test, ARCHITECTURE.md, dashboard, docs/DEMO.md |

### Acceptance criteria (beyond M1, which is detailed in TESTING.md)

- **M2.** Fair policy serves a light tenant ahead of a heavy backlog (scheduler, engine and HTTP
  tests + the `fairness` suite); admission rejects with 503/429 + `Retry-After` on global,
  per-tenant and queued-token limits; cancelled queued requests never reach the model;
  head-of-line blocking quantified for decode-heavy and prefill-heavy workloads.
- **M3.** Batched outputs equal single-request outputs (tiny models: logits within 2e-5; real
  SmolLM2: token-identical greedy outputs vs the Transformers backend); throughput/latency
  sweep vs one-at-a-time baselines at the same model/device/dtype with ≥ 2 rotated rounds.
- **M4.** Prefix-cache hits give outputs identical to cold runs; block accounting invariants
  hold under randomized workloads and preemption; memory is reported (`/metrics`, peak RSS,
  KV usage) and the TTFT effect is measured.
- **M5.** int8 weights measurably smaller with quality (perplexity, top-1 agreement, KL vs fp32)
  reported; external comparison run on the same machine with caveats listed.
- **M6.** Reports include P50/P95/P99 TTFT, TPOT, ITL, E2E, throughput, errors, peak
  RSS/KV/device memory, mean batch size, preemptions, load average.
- **M7.** Readiness fails while loading/draining; SIGTERM mid-stream completes the stream and
  exits; image refuses to start publicly without a key (CI step).

## Explicit non-goals

- Beating vLLM/TGI/llama.cpp; custom CUDA/Metal kernels; tensor/pipeline parallelism;
  speculative decoding; LoRA; multi-node.
- logprobs, `n > 1`, beam search, tool calling, JSON mode.
- Persistence of the prefix cache across restarts; distributed rate limiting.
- Public Internet exposure without a gateway (TLS, real auth).

## Progress log

### 2026-10-14 — M1
- Environment inspected; standalone git repo (the parent home directory is itself a repo).
- Single-request server with SSE, cancellation, deadlines, auth/limits; 55 tests; real-model
  smoke run. Bugs found and fixed: injected backend closed at shutdown; disconnect during an
  uninterruptible step reported as `stop`.

### 2026-10-15 — M2–M7
- **Engine rewrite**: worker-thread step loop; `Scheduler` (continuous batching, decode-first
  chunked prefill, recompute preemption, FCFS/VTC); `BlockManager` (paged blocks, chained-hash
  prefix caching, LRU); native paged-KV Llama/Qwen2/Mistral forward verified against
  Transformers; Transformers `DynamicCache` backend kept as the integration baseline.
- **API**: chat completions, stop strings with held-back streaming, `ignore_eos`, multi-tenant
  keys, Prometheus metrics, `/v1/engine`, `/dashboard`, drain on SIGTERM, JSON logs.
- **Bench**: load generator, A/B suites with rotated order, reports/charts; step profiler;
  int8 quality evaluation; Ollama/vLLM comparison suites.
- **Ops**: Dockerfile (CPU torch via uv index), compose + Prometheus, GitHub Actions.
- **Bugs found and fixed** (each with a regression test): "started" delivered after the first
  step; `max_waiting=0` rejected everything; `DRAIN_TIMEOUT_S=0` meant "wait forever"; dashboard
  counted cached blocks as in use; example script hung on `wait`; no request-body size limit;
  `cp .env.example .env` crashed startup; int8 on MPS emitted only token 0 (capability probe
  used an invalid shape → fp16-overflowing fallback; root-caused by bisecting variants).
- **Benchmarks** run under resource-manager leases on the shared machine; see BENCHMARK.md for
  results, spreads and caveats.

### Remaining risks / next steps
- Single-stream decode is launch-bound (~850–1,250 kernel launches per step); fused kernels
  (Triton/Metal) or graph capture are the real fix; `torch.compile` was evaluated (BENCHMARK.md).
- Gather-based attention materializes K/V per step; a true paged-attention kernel is the main
  gap to vLLM.
- CI and Docker are configured but have not run (repository not pushed).
- Benchmarks were taken on a contended laptop; absolute numbers will differ on a quiet machine,
  relative comparisons were made robust with rotated rounds.
