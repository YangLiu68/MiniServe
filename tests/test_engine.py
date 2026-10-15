"""Engine tests with the deterministic toy backend (simulated step cost via sleeps)."""

import asyncio
import time

import pytest

from miniserve.backends.base import SamplingParams
from miniserve.backends.toy import ToyBackend
from miniserve.engine import (
    Engine,
    EngineClosed,
    EngineConfig,
    EngineOverloaded,
    FinishEvent,
    FinishReason,
    TokenEvent,
)

from conftest import RecordingBackend


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=30))


async def collect(handle):
    tokens, text, finish = [], [], None
    async for ev in handle.events():
        text.append(ev.text)
        if isinstance(ev, FinishEvent):
            finish = ev
        else:
            tokens.append(ev.token_id)
    return tokens, finish, "".join(text)


def P(max_tokens=1000, **kw):
    return SamplingParams(max_tokens=max_tokens, **kw)


async def with_engine(backend, fn, **cfg):
    engine = Engine(backend, EngineConfig(**cfg))
    engine.start()
    try:
        return await fn(engine)
    finally:
        await engine.stop()


def enc(text):
    return list(text.encode())


def test_echo_stop_and_length():
    async def body(engine):
        toks, fin, text = await collect(engine.submit(enc("hello"), P(), 5))
        assert toks == enc("hello") and fin.reason == FinishReason.STOP and text == "hello"
        assert fin.completion_tokens == 5 and fin.prompt_tokens == 5
        toks, fin, _ = await collect(engine.submit(enc("hello"), P(max_tokens=3), 5))
        assert toks == enc("hel") and fin.reason == FinishReason.LENGTH
        assert fin.timings["ttft_ms"] is not None and fin.timings["queue_ms"] is not None

    run(with_engine(ToyBackend(), body))


def test_identical_requests_are_deterministic():
    async def body(engine):
        a = await collect(engine.submit(enc("determinism"), P(), 5))
        b = await collect(engine.submit(enc("determinism"), P(), 5))
        assert a[0] == b[0] and a[1].reason == b[1].reason

    run(with_engine(ToyBackend(), body))


def test_requests_are_batched_concurrently():
    # 20 ms per step regardless of batch size (memory-bound regime): batching must overlap them
    backend = RecordingBackend(ToyBackend(step_overhead_ms=20))

    async def body(engine):
        t0 = time.monotonic()
        hs = [engine.submit(enc(f"request-{i:02d}" * 2), P(), 30) for i in range(4)]
        results = await asyncio.gather(*(collect(h) for h in hs))
        elapsed = time.monotonic() - t0
        assert all(fin.reason == FinishReason.STOP for _, fin, _ in results)
        assert backend.max_batch == 4
        # 21 steps of ~20 ms when batched vs ~4 x 21 sequentially
        assert elapsed < 2 * 21 * 0.020 + 0.3, elapsed

    run(with_engine(backend, body, max_num_seqs=8))


def test_max_num_seqs_1_serializes_fifo():
    backend = RecordingBackend(ToyBackend(decode_ms_per_token=2))

    async def body(engine):
        hs = [engine.submit(enc(f"req{i}" * 3), P(), 10) for i in range(3)]
        await asyncio.gather(*(collect(h) for h in hs))
        spans = [(h.timings.started, h.timings.finished) for h in hs]
        for (_, end_prev), (start_next, _) in zip(spans, spans[1:], strict=False):
            assert start_next >= end_prev - 0.01  # never two requests in the model at once
        assert [s for s, _ in spans] == sorted(s for s, _ in spans)
        assert backend.max_batch == 1

    run(with_engine(backend, body, max_num_seqs=1))


def test_cancel_running_request_stops_within_one_step():
    backend = RecordingBackend(ToyBackend(decode_ms_per_token=10))

    async def body(engine):
        h = engine.submit(enc("x" * 500), P(), 30)
        seen = 0
        async for ev in h.events():
            if isinstance(ev, FinishEvent):
                fin = ev
                break
            seen += 1
            if seen == 5:
                assert h.cancel()
        assert fin.reason == FinishReason.CANCELLED
        await asyncio.sleep(0.1)  # give a misbehaving worker time to keep going
        assert backend.produced <= 6, backend.produced  # at most the step already in flight
        _, fin2, _ = await collect(engine.submit(enc("ok"), P(), 5))
        assert fin2.reason == FinishReason.STOP

    run(with_engine(backend, body))


def test_cancel_queued_request_never_reaches_the_model():
    backend = RecordingBackend(ToyBackend(decode_ms_per_token=5))

    async def body(engine):
        first = engine.submit(enc("a" * 40), P(), 10)
        queued = engine.submit(enc("never"), P(), 10)
        assert queued.state == "queued"
        assert queued.cancel()
        toks, fin, _ = await collect(queued)
        assert fin.reason == FinishReason.CANCELLED and toks == []
        await collect(first)
        await asyncio.sleep(0.05)
        assert queued.id not in backend.seen
        assert not queued.cancel()  # idempotent once finished

    run(with_engine(backend, body, max_num_seqs=1))


def test_timeout_while_running_reports_partial_tokens():
    async def body(engine):
        t0 = time.monotonic()
        toks, fin, _ = await collect(engine.submit(enc("y" * 200), P(), timeout_s=0.25))
        assert fin.reason == FinishReason.TIMEOUT
        assert 3 <= len(toks) <= 15 and fin.completion_tokens == len(toks)
        assert time.monotonic() - t0 < 0.6

    run(with_engine(ToyBackend(decode_ms_per_token=20), body))


def test_timeout_while_queued_counts_queue_wait():
    backend = RecordingBackend(ToyBackend(decode_ms_per_token=10))

    async def body(engine):
        blocker = engine.submit(enc("b" * 50), P(), 10)  # ~0.5 s
        waiting = engine.submit(enc("late"), P(), timeout_s=0.1)
        toks, fin, _ = await collect(waiting)
        assert fin.reason == FinishReason.TIMEOUT and toks == []
        assert fin.timings["queue_ms"] is None  # never started
        await collect(blocker)
        await asyncio.sleep(0.05)
        assert waiting.id not in backend.seen

    run(with_engine(backend, body, max_num_seqs=1))


def test_consumer_deadline_fires_during_a_long_step():
    # One step takes 0.6 s and cannot be interrupted; the client must still get its timeout at
    # ~0.15 s rather than waiting for the step to finish.
    async def body(engine):
        t0 = time.monotonic()
        _, fin, _ = await collect(engine.submit(enc("sixsix"), P(), timeout_s=0.15))
        assert fin.reason == FinishReason.TIMEOUT and time.monotonic() - t0 < 0.4
        _, fin2, _ = await collect(engine.submit(enc("z"), P(), 5))
        assert fin2.reason == FinishReason.STOP

    run(with_engine(ToyBackend(prefill_ms_per_token=100), body))


def test_step_finishing_after_cancel_is_discarded():
    # Regression (found on the real model): the client hung up during a long prefill whose
    # sampled token happened to be EOS, and the request was reported as "stop".
    async def body(engine):
        h = engine.submit(enc("abcd"), P(), 10)  # first step: 4 x 50 ms
        await asyncio.sleep(0.05)
        assert h.state == "running" and h.cancel()
        toks, fin, _ = await collect(h)
        assert fin.reason == FinishReason.CANCELLED and toks == []

    run(with_engine(ToyBackend(prefill_ms_per_token=50), body))


def test_admission_limits():
    async def body(engine):
        running = engine.submit(enc("q" * 50), P(), 10)
        await asyncio.sleep(0.05)  # let it start: running requests do not count as waiting
        a = engine.submit(enc("a"), P(), 10, tenant="t1")
        with pytest.raises(EngineOverloaded) as exc:
            engine.submit(enc("b"), P(), 10, tenant="t1")
        assert exc.value.per_tenant  # t1 hit its own limit first
        b = engine.submit(enc("c"), P(), 10, tenant="t2")
        with pytest.raises(EngineOverloaded) as exc:
            engine.submit(enc("d"), P(), 10, tenant="t3")
        assert not exc.value.per_tenant  # global queue full
        a.cancel()  # cancelling frees capacity immediately
        c = engine.submit(enc("e"), P(), 10, tenant="t3")
        for h in (running, b, c):
            await collect(h)

    run(
        with_engine(
            ToyBackend(decode_ms_per_token=5),
            body,
            max_num_seqs=1,
            max_waiting=2,
            max_waiting_per_tenant=1,
        )
    )


def test_queued_token_budget():
    async def body(engine):
        engine.submit(enc("x" * 60), P(), 10)
        await asyncio.sleep(0.05)
        engine.submit(enc("y" * 60), P(), 10)
        with pytest.raises(EngineOverloaded, match="token budget"):
            engine.submit(enc("z" * 60), P(), 10)

    run(with_engine(ToyBackend(decode_ms_per_token=5), body, max_num_seqs=1, max_queued_tokens=100))


def test_backend_error_is_reported_and_engine_survives():
    class Exploding(ToyBackend):
        def execute(self, batch):
            if any(s.seq.prompt_ids[0] == ord("!") for s in batch):
                raise RuntimeError("boom")
            return super().execute(batch)

    async def body(engine):
        _, fin, _ = await collect(engine.submit(enc("!bad"), P(), 5))
        assert fin.reason == FinishReason.ERROR and "boom" in fin.error
        _, fin, _ = await collect(engine.submit(enc("good"), P(), 5))
        assert fin.reason == FinishReason.STOP

    run(with_engine(Exploding(), body))


def test_stop_cancels_everything_and_rejects_new_work():
    async def body():
        engine = Engine(ToyBackend(decode_ms_per_token=10), EngineConfig(max_num_seqs=1))
        engine.start()
        running = engine.submit(enc("s" * 100), P(), 10)
        queued = engine.submit(enc("s" * 100), P(), 10)
        await asyncio.sleep(0.05)
        await engine.stop()
        await asyncio.sleep(0)  # deliver the worker's final events
        assert running.finish.reason == FinishReason.CANCELLED
        assert queued.finish.reason == FinishReason.CANCELLED
        with pytest.raises(EngineClosed):
            engine.submit(enc("s"), P(), 10)
        assert engine.block_manager.num_used_blocks == 0

    run(body())


def test_stop_strings_are_never_partially_streamed():
    async def body(engine):
        h = engine.submit(enc("hello STOX STOP world"), P(stop=("STOP", "zz")), 5)
        deltas, fin = [], None
        async for ev in h.events():
            if isinstance(ev, TokenEvent):
                deltas.append(ev.text)
            else:
                fin = ev
        assert "".join(deltas) + fin.text == "hello STOX "
        assert fin.reason == FinishReason.STOP
        # "STO" was held back while it could still become "STOP", then released as "STOX"
        assert "S" not in deltas and "STOX" in "".join(deltas)

    run(with_engine(ToyBackend(), body))


def test_ignore_eos_runs_to_max_tokens():
    async def body(engine):
        toks, fin, text = await collect(
            engine.submit(enc("ab"), P(max_tokens=5, ignore_eos=True), 5)
        )
        assert fin.reason == FinishReason.LENGTH and len(toks) == 5 and text == "ab"

    run(with_engine(ToyBackend(), body))


def test_fair_policy_through_the_engine():
    async def order(policy):
        backend = ToyBackend(decode_ms_per_token=2)

        async def body(engine):
            heavy = [engine.submit(enc("h" * 10), P(), 10, tenant="heavy") for _ in range(5)]
            await asyncio.sleep(0.01)
            light = engine.submit(enc("l" * 10), P(), 10, tenant="light")
            hs = heavy + [light]
            await asyncio.gather(*(collect(h) for h in hs))
            return [h.tenant for h in sorted(hs, key=lambda h: h.timings.finished)]

        return await with_engine(backend, body, max_num_seqs=1, policy=policy)

    assert run(order("fcfs"))[-1] == "light"
    assert run(order("fair")).index("light") <= 2


def test_preemption_through_the_engine_keeps_outputs_correct():
    backend = ToyBackend(num_kv_blocks=8, block_size=4, decode_ms_per_token=1)

    async def body(engine):
        prompts = [f"{c}" * 12 for c in "abcd"]
        hs = [engine.submit(enc(p), P(ignore_eos=True, max_tokens=16), 10) for p in prompts]
        results = await asyncio.gather(*(collect(h) for h in hs))
        assert any(fin.preemptions > 0 for _, fin, _ in results)
        for p, (toks, fin, _) in zip(prompts, results, strict=True):
            assert fin.reason == FinishReason.LENGTH
            assert toks == enc(p) + [256] * 4  # echo then EOS (kept because ignore_eos)
        assert engine.block_manager.num_used_blocks == 0

    run(with_engine(backend, body, max_num_batched_tokens=64))


def test_prefix_cache_is_reported_in_finish_event():
    async def body(engine):
        system = "You are a helpful assistant. " * 2  # 58 bytes = 3 full blocks of 16
        await collect(engine.submit(enc(system + "Q1"), P(max_tokens=1), 5))
        _, fin, _ = await collect(engine.submit(enc(system + "Q2"), P(max_tokens=1), 5))
        assert fin.cached_tokens == 48

    run(with_engine(ToyBackend(), body))


def test_request_too_large_for_kv_cache_is_flagged():
    engine = Engine(ToyBackend(num_kv_blocks=4, block_size=4), EngineConfig())
    assert engine.check_request_fits(10, 6) is None
    assert "KV cache capacity" in engine.check_request_fits(10, 7)
    nochunk = Engine(
        ToyBackend(), EngineConfig(enable_chunked_prefill=False, max_num_batched_tokens=32)
    )
    assert "chunked prefill disabled" in nochunk.check_request_fits(30, 3)
