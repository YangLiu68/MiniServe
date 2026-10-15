"""Generation engine (M1): one model, one worker thread, a bounded FIFO of waiting requests.

Threading model
---------------
* The asyncio event loop owns all request state. ``RequestHandle`` fields are only mutated on the
  loop thread (the worker posts events back with ``call_soon_threadsafe``).
* A single worker thread runs the backend's blocking token iterator, so the model is never used
  concurrently. Between tokens it checks the request's cancel flag (a ``threading.Event``) and
  deadline, so cancellation/timeouts take effect within one decode step. A forward pass that is
  already running cannot be interrupted (documented limitation; prefill of a long prompt is the
  worst case).
* The consumer side (``RequestHandle.events``) enforces the deadline as well, so a client gets its
  timeout on time even if the worker is stuck inside a long forward pass.

M2 will replace the FIFO with a fair scheduler; M3 will replace "one request at a time" with
iteration-level batching. The handle/event interface is meant to survive both.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from miniserve.backends.base import Backend, SamplingParams
from miniserve.detokenizer import IncrementalDetokenizer

log = logging.getLogger(__name__)


class FinishReason(StrEnum):
    STOP = "stop"  # EOS sampled
    LENGTH = "length"  # max_tokens reached
    CANCELLED = "cancelled"  # client disconnect, cancel endpoint, or shutdown
    TIMEOUT = "timeout"  # request deadline passed (queue wait + generation)
    ERROR = "error"  # backend raised


class EngineOverloaded(Exception):
    pass


class EngineClosed(Exception):
    pass


@dataclass
class TokenEvent:
    index: int  # 0-based position in the completion
    token_id: int
    text: str  # newly stable text; may be "" while a multi-byte character is incomplete
    t: float  # time.monotonic() when the worker produced it


@dataclass
class Timings:
    arrival: float
    started: float | None = None
    first_token: float | None = None
    last_token: float | None = None
    finished: float | None = None

    def to_dict(self, completion_tokens: int) -> dict[str, float | None]:
        def ms(t: float | None) -> float | None:
            return None if t is None else round((t - self.arrival) * 1000, 3)

        tps = None
        if completion_tokens >= 2 and self.first_token and self.last_token > self.first_token:
            tps = round((completion_tokens - 1) / (self.last_token - self.first_token), 2)
        return {
            "queue_ms": ms(self.started),
            "ttft_ms": ms(self.first_token),
            "total_ms": ms(self.finished),
            "decode_tokens_per_s": tps,
        }


@dataclass
class FinishEvent:
    reason: FinishReason
    prompt_tokens: int
    completion_tokens: int
    text: str = ""  # detokenizer tail flushed at the end
    error: str | None = None
    timings: dict[str, float | None] = field(default_factory=dict)


Event = TokenEvent | FinishEvent


class RequestHandle:
    def __init__(
        self,
        prompt_ids: list[int],
        params: SamplingParams,
        timeout_s: float,
        on_finish: Callable[[RequestHandle], None],
        request_id: str | None = None,
    ) -> None:
        self.id = request_id or f"cmpl-{uuid.uuid4().hex[:24]}"
        self.prompt_ids = prompt_ids
        self.params = params
        self.timings = Timings(arrival=time.monotonic())
        self.deadline = self.timings.arrival + timeout_s
        self.state: Literal["queued", "running", "finished"] = "queued"
        self.finish: FinishEvent | None = None
        self.completion_tokens = 0
        self._cancel_flag = threading.Event()  # read by the worker thread
        self._cancel_reason = FinishReason.CANCELLED
        self._events: asyncio.Queue[Event] = asyncio.Queue()
        self._on_finish = on_finish

    # ---- loop-thread API ----
    def cancel(self, reason: FinishReason = FinishReason.CANCELLED) -> bool:
        """Request cancellation. Returns False if the request already finished."""
        if self.finish is not None:
            return False
        self._cancel_reason = reason
        self._cancel_flag.set()
        if self.state == "queued":  # nothing running: finish right away
            self._push(self._make_finish(reason))
        return True

    async def events(self) -> AsyncIterator[Event]:
        """Yield token events then exactly one FinishEvent; enforces the deadline."""
        while True:
            if not self._events.empty():
                ev = self._events.get_nowait()
            else:
                remaining = self.deadline - time.monotonic()
                if remaining <= 0:
                    self._expire()
                    continue
                try:
                    ev = await asyncio.wait_for(self._events.get(), timeout=remaining)
                except TimeoutError:
                    self._expire()
                    continue
            yield ev
            if isinstance(ev, FinishEvent):
                return

    # ---- internals ----
    def _expire(self) -> None:
        if self.finish is None:
            self._cancel_reason = FinishReason.TIMEOUT
            self._cancel_flag.set()
            self._push(self._make_finish(FinishReason.TIMEOUT))

    def _make_finish(self, reason: FinishReason, text: str = "", error: str | None = None):
        return FinishEvent(reason, len(self.prompt_ids), self.completion_tokens, text, error)

    def _push(self, ev: Event) -> None:
        if self.finish is not None:
            return  # late events from the worker after a consumer-side timeout/cancel
        if isinstance(ev, TokenEvent):
            self.completion_tokens += 1
            if self.timings.first_token is None:
                self.timings.first_token = ev.t
            self.timings.last_token = ev.t
        else:
            self.timings.finished = time.monotonic()
            ev.completion_tokens = self.completion_tokens
            ev.timings = self.timings.to_dict(self.completion_tokens)
            self.finish = ev
            self.state = "finished"
            self._on_finish(self)
        self._events.put_nowait(ev)


class Engine:
    def __init__(self, backend: Backend, max_pending: int) -> None:
        self.backend = backend
        self.max_pending = max_pending
        self._queue: asyncio.Queue[RequestHandle] = asyncio.Queue()
        self._active: dict[str, RequestHandle] = {}  # queued + running
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="miniserve-gen")
        self._task: asyncio.Task | None = None
        self._closing = False
        self.current: RequestHandle | None = None

    # ---- lifecycle ----
    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="miniserve-engine")

    async def stop(self) -> None:
        """Cancel everything, wait for the worker to leave the model, release the thread."""
        self._closing = True
        for handle in list(self._active.values()):
            handle.cancel()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        await asyncio.to_thread(self._executor.shutdown, wait=True)

    # ---- request API ----
    def submit(
        self, prompt_ids: list[int], params: SamplingParams, timeout_s: float
    ) -> RequestHandle:
        if self._closing:
            raise EngineClosed("engine is shutting down")
        if len(self._active) >= self.max_pending + 1:  # +1: the running slot
            raise EngineOverloaded(f"{len(self._active)} requests in flight")
        handle = RequestHandle(prompt_ids, params, timeout_s, on_finish=self._on_finish)
        self._active[handle.id] = handle
        self._queue.put_nowait(handle)
        return handle

    def get(self, request_id: str) -> RequestHandle | None:
        return self._active.get(request_id)

    @property
    def num_active(self) -> int:
        return len(self._active)

    def _on_finish(self, handle: RequestHandle) -> None:
        self._active.pop(handle.id, None)
        f = handle.finish
        log.info(
            "request finished id=%s reason=%s prompt_tokens=%d completion_tokens=%d timings=%s%s",
            handle.id,
            f.reason,
            f.prompt_tokens,
            f.completion_tokens,
            f.timings,
            f" error={f.error}" if f.error else "",
        )

    # ---- scheduling loop ----
    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            handle = await self._queue.get()
            if handle.finish is not None:  # cancelled while queued
                continue
            if time.monotonic() >= handle.deadline:
                handle._expire()
                continue
            handle.state = "running"
            handle.timings.started = time.monotonic()
            self.current = handle
            try:
                await loop.run_in_executor(self._executor, self._generate, handle, loop)
            finally:
                self.current = None

    def _generate(self, h: RequestHandle, loop: asyncio.AbstractEventLoop) -> None:
        """Worker thread: drive the backend iterator for one request."""

        def push(ev: Event) -> None:
            loop.call_soon_threadsafe(h._push, ev)

        def interrupted() -> FinishReason | None:
            if h._cancel_flag.is_set():
                return h._cancel_reason
            if time.monotonic() >= h.deadline:
                return FinishReason.TIMEOUT
            return None

        tokenizer = self.backend.tokenizer
        detok = IncrementalDetokenizer(tokenizer)
        n = 0
        reason, error, tail = FinishReason.ERROR, None, ""
        it = self.backend.generate(h.prompt_ids, h.params)
        try:
            while True:
                # Before each forward pass: a stopped request costs no extra compute.
                if (stop := interrupted()) is not None:
                    reason = stop
                    break
                try:
                    tok = next(it)
                except StopIteration:
                    reason = FinishReason.STOP
                    break
                # After it: a step that finished after cancel/deadline is discarded, so the
                # client-visible result is "cancelled"/"timeout", never a late token or EOS.
                if (stop := interrupted()) is not None:
                    reason = stop
                    break
                if tok in tokenizer.eos_token_ids:
                    reason = FinishReason.STOP
                    break
                push(TokenEvent(n, tok, detok.add(tok), time.monotonic()))
                n += 1
                if n >= h.params.max_tokens:
                    reason = FinishReason.LENGTH
                    break
            tail = detok.flush()
        except Exception as e:  # backend failure must not kill the engine loop
            log.exception("generation failed id=%s", h.id)
            reason, error = FinishReason.ERROR, f"{type(e).__name__}: {e}"
        finally:
            it.close()
        push(h._make_finish(reason, text=tail, error=error))
