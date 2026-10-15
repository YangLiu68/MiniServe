"""Engine tests with the deterministic toy backend (simulated per-token compute via sleeps)."""

import asyncio
import time

import pytest

from miniserve.backends.base import SamplingParams
from miniserve.backends.toy import ToyBackend
from miniserve.engine import Engine, EngineClosed, EngineOverloaded, FinishEvent, FinishReason

from conftest import RecordingBackend


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=20))


async def collect(handle):
    tokens, finish = [], None
    async for ev in handle.events():
        if isinstance(ev, FinishEvent):
            finish = ev
        else:
            tokens.append(ev.token_id)
    return tokens, finish


def P(max_tokens=1000, **kw):
    return SamplingParams(max_tokens=max_tokens, **kw)


async def with_engine(backend, fn, max_pending=8):
    engine = Engine(backend, max_pending=max_pending)
    engine.start()
    try:
        return await fn(engine)
    finally:
        await engine.stop()


def test_echo_stop_and_length():
    backend = ToyBackend()
    prompt = backend.tokenizer.encode("hello")

    async def body(engine):
        toks, fin = await collect(engine.submit(prompt, P(), 5))
        assert toks == prompt and fin.reason == FinishReason.STOP
        assert fin.completion_tokens == 5 and fin.prompt_tokens == 5
        toks, fin = await collect(engine.submit(prompt, P(max_tokens=3), 5))
        assert toks == prompt[:3] and fin.reason == FinishReason.LENGTH
        assert fin.timings["ttft_ms"] is not None and fin.timings["total_ms"] >= 0

    run(with_engine(backend, body))


def test_identical_requests_are_deterministic():
    backend = ToyBackend()
    prompt = backend.tokenizer.encode("determinism")

    async def body(engine):
        a = await collect(engine.submit(prompt, P(), 5))
        b = await collect(engine.submit(prompt, P(), 5))
        assert a[0] == b[0] and a[1].reason == b[1].reason

    run(with_engine(backend, body))


def test_cancel_running_request_stops_within_one_step():
    backend = RecordingBackend(ToyBackend(decode_ms_per_token=10))
    prompt = backend.tokenizer.encode("x" * 500)

    async def body(engine):
        h = engine.submit(prompt, P(), 30)
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
        # one in-flight step may complete after cancel(); nothing beyond that
        assert backend.yielded <= 6, backend.yielded
        assert fin.completion_tokens <= 6
        # the engine is free again
        toks, fin2 = await collect(engine.submit(backend.tokenizer.encode("ok"), P(), 5))
        assert fin2.reason == FinishReason.STOP

    run(with_engine(backend, body))


def test_cancel_queued_request_never_runs():
    backend = RecordingBackend(ToyBackend(decode_ms_per_token=5))
    long_prompt = backend.tokenizer.encode("a" * 40)

    async def body(engine):
        first = engine.submit(long_prompt, P(), 10)
        queued = engine.submit(backend.tokenizer.encode("never"), P(), 10)
        assert queued.state == "queued"
        assert queued.cancel()
        toks, fin = await collect(queued)
        assert fin.reason == FinishReason.CANCELLED and toks == []
        await collect(first)
        await asyncio.sleep(0.05)
        assert len(backend.calls) == 1  # the cancelled request never reached the model
        assert not queued.cancel()  # idempotent once finished

    run(with_engine(backend, body))


def test_timeout_while_running_reports_partial_tokens():
    backend = ToyBackend(decode_ms_per_token=20)
    prompt = backend.tokenizer.encode("y" * 200)

    async def body(engine):
        t0 = time.monotonic()
        toks, fin = await collect(engine.submit(prompt, P(), timeout_s=0.25))
        elapsed = time.monotonic() - t0
        assert fin.reason == FinishReason.TIMEOUT
        assert 3 <= len(toks) <= 15 and fin.completion_tokens == len(toks)
        assert elapsed < 0.6

    run(with_engine(backend, body))


def test_timeout_while_queued_counts_queue_wait():
    backend = RecordingBackend(ToyBackend(decode_ms_per_token=10))

    async def body(engine):
        blocker = engine.submit(backend.tokenizer.encode("b" * 50), P(), 10)  # ~0.5 s
        waiting = engine.submit(backend.tokenizer.encode("late"), P(), timeout_s=0.1)
        toks, fin = await collect(waiting)
        assert fin.reason == FinishReason.TIMEOUT and toks == []
        assert fin.timings["queue_ms"] is None  # never started
        await collect(blocker)
        await asyncio.sleep(0.05)
        assert len(backend.calls) == 1

    run(with_engine(backend, body))


def test_consumer_deadline_fires_during_a_long_forward_pass():
    # One "prefill" step takes 0.6 s and cannot be interrupted; the client must still get its
    # timeout at ~0.15 s rather than waiting for the step to finish.
    backend = ToyBackend(prefill_ms_per_token=100)
    prompt = backend.tokenizer.encode("sixsix")

    async def body(engine):
        t0 = time.monotonic()
        _, fin = await collect(engine.submit(prompt, P(), timeout_s=0.15))
        assert fin.reason == FinishReason.TIMEOUT
        assert time.monotonic() - t0 < 0.4
        # the engine waits for the worker to leave the model before running the next request
        toks, fin2 = await collect(engine.submit(backend.tokenizer.encode("z"), P(), 5))
        assert fin2.reason == FinishReason.STOP

    run(with_engine(backend, body))


def test_step_finishing_after_cancel_is_discarded():
    # Regression (found on the real model): the client hung up during a 10 s prefill whose
    # sampled token happened to be EOS, and the request was reported as "stop", not "cancelled".
    backend = ToyBackend(prefill_ms_per_token=50)
    prompt = backend.tokenizer.encode("abcd")  # 0.2 s uninterruptible first step

    async def body(engine):
        h = engine.submit(prompt, P(), 10)
        await asyncio.sleep(0.05)
        assert h.state == "running" and h.cancel()
        toks, fin = await collect(h)
        assert fin.reason == FinishReason.CANCELLED and toks == []

    run(with_engine(backend, body))


def test_requests_are_serialized_fifo():
    backend = ToyBackend(decode_ms_per_token=5)

    async def body(engine):
        hs = [engine.submit(backend.tokenizer.encode(f"req{i}" * 3), P(), 10) for i in range(3)]
        await asyncio.gather(*(collect(h) for h in hs))
        spans = [(h.timings.started, h.timings.finished) for h in hs]
        for (_, end_prev), (start_next, _) in zip(spans, spans[1:], strict=False):
            assert start_next >= end_prev - 0.01  # never two requests on the model at once
        assert [s for s, _ in spans] == sorted(s for s, _ in spans)  # FIFO

    run(with_engine(backend, body))


def test_admission_limit_rejects_overflow():
    backend = ToyBackend(decode_ms_per_token=10)
    prompt = backend.tokenizer.encode("q" * 20)

    async def body(engine):
        a = engine.submit(prompt, P(), 10)  # running slot
        b = engine.submit(prompt, P(), 10)  # the one pending slot
        with pytest.raises(EngineOverloaded):
            engine.submit(prompt, P(), 10)
        b.cancel()  # cancelling frees capacity immediately
        c = engine.submit(prompt, P(), 10)
        await collect(a)
        await collect(c)

    run(with_engine(backend, body, max_pending=1))


def test_backend_error_is_reported_and_engine_survives():
    class Exploding(ToyBackend):
        def generate(self, prompt_ids, params):
            yield prompt_ids[0]
            raise RuntimeError("boom")

    backend = Exploding()

    async def body(engine):
        toks, fin = await collect(engine.submit(backend.tokenizer.encode("ab"), P(), 5))
        assert fin.reason == FinishReason.ERROR and "boom" in fin.error and len(toks) == 1
        toks, fin = await collect(engine.submit(backend.tokenizer.encode("cd"), P(), 5))
        assert fin.reason == FinishReason.ERROR  # still processing requests

    run(with_engine(backend, body))


def test_stop_cancels_everything_and_rejects_new_work():
    backend = ToyBackend(decode_ms_per_token=10)
    prompt = backend.tokenizer.encode("s" * 100)

    async def body():
        engine = Engine(backend, max_pending=4)
        engine.start()
        running = engine.submit(prompt, P(), 10)
        queued = engine.submit(prompt, P(), 10)
        await asyncio.sleep(0.05)
        await engine.stop()
        assert running.finish.reason == FinishReason.CANCELLED
        assert queued.finish.reason == FinishReason.CANCELLED
        with pytest.raises(EngineClosed):
            engine.submit(prompt, P(), 10)

    run(body())
