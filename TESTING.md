# Testing

## Commands

```bash
uv sync --extra hf
uv run ruff check src tests && uv run ruff format --check src tests
uv run pytest                                            # default: no downloads
MINISERVE_RUN_REAL_MODEL_TESTS=1 uv run pytest -m real_model   # downloads SmolLM2-135M (~270 MB)
uv run pytest -m "not torch and not real_model"          # toy-only (works without torch)
```

## Test layers

| Layer | File | Backend | What it proves |
|---|---|---|---|
| Unit | `test_detokenizer.py` | byte tokenizer | Multi-byte characters held back until complete; concatenated deltas == full decode (200 random strings); truncated tail flushed |
| Engine | `test_engine.py` | toy (sleep-simulated cost) | Echo/stop/length, determinism, cancel running (≤1 extra step) and queued (never reaches model), timeouts while running / queued / inside an uninterruptible step, discard of a step finishing after cancel, FIFO serialisation, admission limit, backend errors, shutdown |
| HTTP | `test_api.py` | toy | Response shape, exact `max_tokens`, stream == non-stream (incl. emoji/CJK), 422 validation matrix, 400 limits incl. exact context boundary, 404 model, auth, 429 rate limit, loading→ready lifecycle, failed load, 504/in-band timeouts, server timeout cap, bind safety, injected-backend ownership |
| Live socket | `test_live_cancellation.py` | toy on real uvicorn | Real TCP disconnect (stream and non-stream) stops generation — measured as tokens the backend produced; cancel endpoint; 503 overload over HTTP |
| Torch path | `test_hf_backend.py` | tiny random GPT-2 | Manual prefill/decode loop with `DynamicCache` == `model.generate()` greedy (4 prompts × 24 tokens); seeded sampling reproducible and seed-sensitive; tiny top-p == greedy; `sample_next` nucleus set; API stream == non-stream |
| Real model (opt-in) | `test_real_model.py` | SmolLM2-135M-Instruct, CPU fp32 | Sensible greedy output ("Paris"), stream == non-stream, incremental detokenizer == full decode on the real tokenizer (spaces, newlines, emoji, CJK, code) |

## M1 acceptance criteria → evidence

| # | Criterion (PLAN.md) | Evidence |
|---|---|---|
| 1 | JSON + SSE completions, deltas concatenate | `test_stream_matches_non_stream`, `test_api_stream_equals_non_stream_on_tiny_model`, real-model `test_stream_equals_non_stream`; manual curl (docs/DEMO.md) |
| 2 | Background load, `/readyz` 503→200 | `test_not_ready_while_loading_then_ready`, `test_failed_load_is_reported`; observed live with SmolLM2 on MPS |
| 3 | 422 / 400 / 404 validation | `test_validation_errors` (10 cases), `test_limits_return_400`, `test_wrong_model_is_404` |
| 4 | Exact `max_tokens`, EOS → stop | `test_max_tokens_is_enforced_exactly`, `test_echo_stop_and_length` |
| 5 | Disconnect + cancel endpoint stop within one step | `test_live_cancellation.py`; `test_cancel_running_request_stops_within_one_step`; real model: disconnect after 5 of 400 tokens → `cancelled`, 5 tokens produced |
| 6 | Deadlines incl. queue wait, in-band/504 | `test_timeout_*`, `test_consumer_deadline_fires_during_a_long_forward_pass`, `test_server_timeout_caps_client_timeout` |
| 7 | No concurrent model use; bounded FIFO; 503 overflow | `test_requests_are_serialized_fifo`, `test_admission_limit_rejects_overflow`, `test_overload_returns_503_over_http` |
| 8 | Bind safety, auth, rate limit | `test_bind_safety`, `test_auth_required_when_key_set`, `test_rate_limit_returns_429_with_retry_after`; CLI exits 2 on `MINISERVE_HOST=0.0.0.0` without key (run manually) |
| 9 | Deterministic toy tests; loop == `generate()`; seeded sampling | `test_engine.py`, `test_hf_backend.py` |
| 10 | Real-model end-to-end via HTTP | Ran `miniserve` (SmolLM2, MPS fp16) + curl: auth, JSON, SSE, disconnect, SIGTERM shutdown (docs/DEMO.md) |

## Are the tests sensitive? (mutation checks actually run)

- Removed all three disconnect→cancel paths in `server.py` → both live disconnect tests **failed**;
  restored → pass.
- Removed the engine's post-step cancel check → `test_step_finishing_after_cancel_is_discarded`
  **failed**; restored → pass.

## Latest results (2026-10-14, Apple M4, macOS 15.3.2, Python 3.12.10)

```
ruff check / format --check      : clean
pytest (default)                 : 55 passed, 3 skipped (real-model, opt-in)
pytest -m real_model (opt-in)    : 3 passed
```

## Bugs found by testing during M1

1. **App closed an injected backend on shutdown** (found by the real-model module fixture being
   reused across two app lifecycles). Fix: only close backends the app loaded itself. Regression:
   `test_injected_backend_survives_app_shutdown`.
2. **Disconnect during an uninterruptible step was reported as `stop`** (found in a real-model
   run: client hung up during a ~10 s prefill whose sampled token was EOS). Fix: re-check
   cancel/deadline after each step and discard the step's result. Regression:
   `test_step_finishing_after_cancel_is_discarded`; re-verified on the real model.

## Environment notes / flakiness risks

- On the development machine, imports are I/O-bound and slow (FastAPI 3–12 s, torch +
  transformers ~10 s wall at low CPU) while the host was heavily loaded by other processes (load
  average 33–121 on 10 cores). Suite wall time (30–80 s) is dominated by that, not by test bodies
  (slowest test body ≈ 1 s).
- Timing-based tests use generous margins (e.g. "fewer than 60 of 300 tokens", "< 0.6 s"), but a
  severely starved CI runner could still flake them; they are the first suspects if CI is red.
- CI (GitHub Actions) is planned for M7 and has **not** run yet.
