# Testing

```bash
make install            # uv sync --extra hf --extra bench
make lint               # ruff check + format --check
make test               # 126 tests: toy + tiny random models, no downloads, CPU only
make test-real          # +4 tests on the real SmolLM2-135M (downloads ~270 MB)
uv run pytest -m "not torch and not real_model"   # what CI runs without torch installed
```

## Strategy

The engine is split so that most of it can be tested **without a model**: the scheduler and
block manager are pure Python driven by a fake model; the engine is driven by a deterministic
toy backend whose step cost is a controlled sleep; the numerics are tested by comparing our
forward pass against Transformers on tiny random models (no download), and finally on the real
model (opt-in). Timing-sensitive behaviour (cancellation, deadlines, drain) is tested against
real uvicorn servers, real sockets and a real SIGTERM, and asserts on **compute actually spent**
(tokens the backend produced), not just on response codes.

| Layer | File | Tests | What it proves |
|---|---|---:|---|
| Block allocator | `test_block_manager.py` | 10 | block accounting, prefix sharing + ref counts, chained hashes don't alias, ≥1 token always recomputed, LRU evicts tails first, revival accounting, **600-step randomized run with invariants checked after every op** |
| Scheduler | `test_scheduler.py` | 11 | continuous batching (join mid-flight), token budget + decode-first chunked prefill, no-chunk mode, `max_num_seqs`, recompute preemption of the youngest (outputs still exact), preempted seqs re-admitted first, prefix hits schedule only the tail, FCFS vs VTC order, no idle-credit banking, recompute not billed, **randomized simulation with cancellations: no deadlock, no leaked blocks** |
| Stop strings | `test_stop.py` | 4 | split-across-deltas, false alarms released, earliest stop wins, **2,000 randomized cases vs a reference truncation** |
| Detokenizer | `test_detokenizer.py` | 3 | multi-byte held back, concatenation == full decode (200 random strings), truncated tail flushed |
| Engine | `test_engine.py` | 20 | batching overlaps requests, `max_num_seqs=1` serializes FIFO, cancel running (≤1 extra step) / queued (never reaches model), timeouts (running, queued, inside an uninterruptible step), late step discarded, admission limits (global, per tenant, queued tokens), backend errors isolated, shutdown, stop strings never partially streamed, `ignore_eos`, fairness, preemption with exact outputs, prefix-cache reporting, KV-capacity validation |
| HTTP API | `test_api.py` | 40 | response shapes, exact `max_tokens`, stream == non-stream (emoji/CJK), 13-case validation matrix, 400 limits incl. exact boundaries, auth, multi-tenant keys + per-tenant rate limits, loading → ready, failed load, 504 / in-band timeouts, chat (JSON + SSE, role chunk), `stop` / `ignore_eos`, `/metrics` contents, `/v1/engine`, draining, injected-backend ownership |
| Live sockets | `test_live_cancellation.py` | 4 | real TCP disconnect (stream and non-stream) stops generation (measured in produced tokens), cancel endpoint, 503 overload |
| Process | `test_drain.py` | 2 | **SIGTERM mid-stream: the stream completes, the process drains and exits**; refuses public bind without a key |
| Native numerics | `test_native.py` | 22 | for Llama, Qwen2 (QKV bias) and Mistral tiny models: full-prefill logits == HF (atol 2e-5), chunked prefill (1/3/7) == full, **mixed ragged batch == individual HF runs**, greedy == HF reference, prefix-cache hit == cold, preemption under memory pressure == unconstrained; seeded sampling independent of batch composition; int8 close to fp (cos > 0.999) and smaller; fused int8 kernel == dequant fallback; unsupported archs rejected |
| HF backend | `test_hf_backend.py` | 4 | generic `DynamicCache` path on GPT-2 (non-Llama) == reference, preemption rebuilds caches, sampler unit tests, seeded API reproducibility |
| Bench tooling | `test_bench.py` | 6 | percentile convention, warmup/error handling in summaries, exact-length prompts, deterministic workloads, Prometheus parsing, end-to-end multi-tenant run against a live server |
| Real model (opt-in) | `test_real_model.py` | 4 | **native engine (batched + chunked) greedy output == Transformers backend, token for token, on SmolLM2-135M**; sensible completion and chat ("Paris"); prefix-cache hit on real prompts; incremental detokenizer == full decode on the real tokenizer |

## Are the tests sensitive? (mutation checks actually run)

| Mutation | Result |
|---|---|
| Remove the three disconnect → cancel paths in `server.py` | both live disconnect tests fail |
| Remove the engine's post-step cancellation check | `test_step_finishing_after_cancel_is_discarded` fails |
| Remove the causal mask in `build_metadata` | 6 native tests fail (logits, chunked, mixed batch, greedy, prefix, preemption) |
| Restore `DRAIN_TIMEOUT_S` → `None` conversion | zero-drain SIGTERM test fails |
| Restore the 8×32 int8 probe (MPS) | probe/dispatch test fails |
| Remove the fp32 upcast in the int8 fallback | forced-fallback fp16 overflow test fails |

## Bugs found by testing (all fixed, each with a regression test)

| # | Bug | How it was found |
|---|---|---|
| 1 | App closed an **injected** backend at shutdown | real-model fixture reused across two apps |
| 2 | Disconnect during an uninterruptible step whose token was EOS reported as `stop` | real-model run (client hung up mid-prefill) |
| 3 | "Request started" delivered *after* the first step, so a cancel during the first prefill took the "queued" path | engine test |
| 4 | `max_waiting=0` rejected every request even when idle | live overload test |
| 5 | `MINISERVE_DRAIN_TIMEOUT_S=0` passed to uvicorn as `None` = wait forever | code review → real-SIGTERM test |
| 6 | Dashboard counted cached blocks as "in use" (`"c" >= "1"` is true in JS) | headless-Chrome screenshot |
| 7 | Example script hung forever (bare `wait` also waited on the server) and didn't show which tenant was which | running the example |
| 8 | No request-body size limit: a huge body is buffered before validation | code review → 413 middleware tests |
| 9 | `cp .env.example .env` (as documented) crashed startup on empty values | test that loads the example as an env file |
| 10 | **int8 on MPS produced only token 0.** The capability probe used an 8×32 weight; the MPS kernel needs N, K multiples of 32, so MPS was marked unsupported and every call took the dequantize fallback, which multiplied by raw int8 values before scaling and overflowed fp16 → NaN logits | quant benchmark (empty text) → bisect of the original code vs single-change variants on the exact benchmark prompts; two plausible hypotheses (fused-kernel overflow, fp16-subnormal scales) were tested and refuted first |

A test-helper pitfall worth remembering: `policy or FCFSPolicy()` silently replaced an *empty*
`FairPolicy` (it defines `__len__`, so it is falsy).

## Latest results

| Date | Command | Result |
|---|---|---|
| 2026-10-15 | `make lint` | clean |
| 2026-10-15 | `make test` | 126 passed, 4 skipped (real-model, opt-in) |
| 2026-10-15 | `make test-real` | 4 passed (CPU fp32) |

Host: Apple M4 (10 cores, 16 GB), macOS 15.3.2, Python 3.12.10, torch 2.14.1, transformers
5.19.0. GitHub Actions is configured (lint; tests on 3.11 + 3.12 with CPU torch; torch-free core;
Docker build + container smoke test) but **has not run yet** — the repository has not been
pushed.

## Flakiness notes

* Timing-based tests use generous margins, but a severely starved runner could still flake
  them; they are the first suspects if CI is red.
* On the development machine imports are I/O-bound and slow (several seconds for
  FastAPI/torch) because the machine is shared with other heavy jobs; suite wall time is
  dominated by that, not by test bodies.
