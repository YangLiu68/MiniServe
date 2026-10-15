"""Generation engine: asyncio front end + a worker thread running the continuous-batching loop.

Threading model
---------------
* **Event-loop thread** owns ``RequestHandle`` objects (what HTTP handlers see): admission
  control, cancellation requests, the per-request event queue and the consumer-side deadline.
* **Worker thread** owns everything else — ``Scheduler``, ``BlockManager``, ``Sequence`` state and
  the backend (model + KV cache) — and runs::

      drain inbox -> reap cancelled/expired -> schedule() -> backend.execute(batch)
      -> append tokens, detokenize, check stop conditions -> post events back to the loop

* The two sides communicate only through a locked inbox (new sequences), a per-request
  ``threading.Event`` (cancel flag), and ``loop.call_soon_threadsafe`` (one batch of events per
  step). No asyncio object is touched from the worker; no scheduler state from the loop.

With speculative decoding a step can produce several tokens for one sequence (verified draft
tokens + one sampled token); they are delivered in order and stop conditions are checked after
each one, so a stop string, EOS or ``max_tokens`` inside an accepted draft ends the request
exactly where non-speculative decoding would have.

Cancellation and deadlines are checked before each step (a cancelled sequence is not scheduled)
and again after it (a token computed after cancel/deadline is discarded, never delivered).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from collections import Counter
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from miniserve.backends.base import Backend, SamplingParams
from miniserve.block_manager import BlockManager
from miniserve.detokenizer import IncrementalDetokenizer
from miniserve.policy import make_policy
from miniserve.scheduler import Scheduler, SchedulerConfig, SchedulerOutput
from miniserve.sequence import SeqStatus, Sequence
from miniserve.speculative import NgramProposer
from miniserve.stop import StopChecker

log = logging.getLogger(__name__)


class FinishReason(StrEnum):
    STOP = "stop"  # EOS sampled or a stop string matched
    LENGTH = "length"  # max_tokens reached
    CANCELLED = "cancelled"  # client disconnect, cancel endpoint, or shutdown
    TIMEOUT = "timeout"  # request deadline passed (queue wait + generation)
    ERROR = "error"  # backend raised


class EngineOverloaded(Exception):
    def __init__(self, message: str, per_tenant: bool = False) -> None:
        super().__init__(message)
        self.per_tenant = per_tenant


class EngineClosed(Exception):
    pass


@dataclass
class EngineConfig:
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 512
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    policy: str = "fcfs"
    max_waiting: int = 256  # requests allowed to wait beyond the max_num_seqs running slots
    max_waiting_per_tenant: int = 64
    max_queued_tokens: int = 1 << 20  # prompt tokens waiting (bounds queue memory)
    watermark: float = 0.01  # fraction of KV blocks kept free when admitting
    speculative: str = "none"  # "ngram": prompt-lookup speculative decoding
    num_speculative_tokens: int = 4
    ngram_max: int = 3
    ngram_min: int = 2
    spec_max_batch: int = 4  # speculate only while at most this many sequences decode
    # "tenant": prefix-cache blocks are only shared within a tenant (no cross-tenant timing or
    # cached_tokens side channel about other tenants' prompts); "global": shared by everyone
    prefix_cache_scope: str = "global"


@dataclass
class TokenEvent:
    index: int  # 0-based position in the completion
    token_id: int
    text: str  # newly stable text; may be "" (incomplete UTF-8 or held back for stop strings)
    t: float  # time.monotonic() when the worker produced it


@dataclass
class _Started:
    t: float


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
    completion_tokens: int = 0
    text: str = ""  # tail flushed at the end
    error: str | None = None
    timings: dict[str, float | None] = field(default_factory=dict)
    cached_tokens: int = 0
    preemptions: int = 0
    draft_tokens: int = 0  # speculative decoding: proposed ...
    accepted_tokens: int = 0  # ... and accepted draft tokens


Event = TokenEvent | FinishEvent


class RequestHandle:
    """Loop-thread view of one request."""

    def __init__(
        self,
        prompt_ids: list[int],
        params: SamplingParams,
        timeout_s: float,
        tenant: str,
        on_finish: Callable[[RequestHandle], None],
        wake: Callable[[], None],
    ) -> None:
        self.id = f"cmpl-{uuid.uuid4().hex[:24]}"
        self.prompt_ids = prompt_ids
        self.params = params
        self.tenant = tenant
        self.timings = Timings(arrival=time.monotonic())
        self.deadline = self.timings.arrival + timeout_s
        self.state: Literal["queued", "running", "finished"] = "queued"
        self.finish: FinishEvent | None = None
        self.completion_tokens = 0
        self._cancel_flag = threading.Event()  # read by the worker thread
        self._cancel_reason = FinishReason.CANCELLED
        self._events: asyncio.Queue[Event] = asyncio.Queue()
        self._on_finish = on_finish
        self._wake = wake

    def cancel(self, reason: FinishReason = FinishReason.CANCELLED) -> bool:
        """Request cancellation. Returns False if the request already finished."""
        if self.finish is not None:
            return False
        self._cancel_reason = reason
        self._cancel_flag.set()
        if self.state == "queued":  # not running yet: answer immediately
            self._push(FinishEvent(reason, len(self.prompt_ids)))
        self._wake()  # let the worker free its resources promptly
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

    def _expire(self) -> None:
        if self.finish is None:
            self._cancel_reason = FinishReason.TIMEOUT
            self._cancel_flag.set()
            self._push(FinishEvent(FinishReason.TIMEOUT, len(self.prompt_ids)))
            self._wake()

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
            self._on_finish(self)
            self.state = "finished"
        self._events.put_nowait(ev)


class Engine:
    def __init__(
        self, backend: Backend, config: EngineConfig | None = None, metrics: Any = None
    ) -> None:
        self.backend = backend
        self.config = cfg = config or EngineConfig()
        self.metrics = metrics
        prefix = cfg.enable_prefix_caching and backend.supports_prefix_caching
        self.block_manager = BlockManager(backend.num_kv_blocks, backend.block_size, prefix)
        self.proposer: NgramProposer | None = None
        if cfg.speculative == "ngram":
            if getattr(backend, "supports_speculation", False):
                self.proposer = NgramProposer(
                    cfg.num_speculative_tokens, cfg.ngram_max, cfg.ngram_min
                )
            else:
                log.warning("backend %r cannot verify drafts; speculation disabled", backend.name)
        elif cfg.speculative != "none":
            raise ValueError(f"unknown speculative method {cfg.speculative!r}")
        self.scheduler = Scheduler(
            SchedulerConfig(
                max_num_seqs=cfg.max_num_seqs,
                max_num_batched_tokens=cfg.max_num_batched_tokens,
                enable_chunked_prefill=cfg.enable_chunked_prefill,
                watermark_blocks=int(cfg.watermark * backend.num_kv_blocks),
                spec_max_batch=cfg.spec_max_batch,
            ),
            self.block_manager,
            make_policy(cfg.policy),
            on_release=backend.release,
            proposer=self.proposer,
        )
        # worker side
        self._inbox: list[Sequence] = []
        self._inbox_lock = threading.Lock()
        self._wake_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stopping = False
        self.num_steps = 0
        self.spec_steps = 0  # steps that verified at least one draft token
        self.spec_draft_tokens = 0
        self.spec_accepted_tokens = 0
        self.fatal_error: str | None = None
        # loop side
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closing = False
        self._active: dict[str, RequestHandle] = {}
        self._queued: set[str] = set()
        self._tenant_queued: Counter[str] = Counter()
        self._queued_tokens = 0

    # ------------------------------------------------------------------ capacity / validation
    @property
    def capacity_tokens(self) -> int:
        return self.block_manager.capacity_tokens

    def check_request_fits(self, prompt_tokens: int, max_tokens: int) -> str | None:
        """Reasons a request can never be served by this engine configuration, else None."""
        total = prompt_tokens + max_tokens
        if total > self.capacity_tokens:
            return (
                f"prompt ({prompt_tokens}) + max_tokens ({max_tokens}) exceeds the KV cache "
                f"capacity ({self.capacity_tokens} tokens)"
            )
        if not self.config.enable_chunked_prefill and total > self.config.max_num_batched_tokens:
            # a preempted sequence is recomputed in one step without chunking
            return (
                f"prompt + max_tokens ({total}) exceeds max_num_batched_tokens "
                f"({self.config.max_num_batched_tokens}) with chunked prefill disabled"
            )
        return None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._thread = threading.Thread(
            target=self._worker_main, name="miniserve-engine", daemon=True
        )
        self._thread.start()

    async def stop(self) -> None:
        """Cancel everything, wait for the worker to leave the model, end the thread."""
        self._closing = True
        for handle in list(self._active.values()):
            handle.cancel()
        self._stopping = True
        self._wake_event.set()
        if self._thread is not None:
            await asyncio.to_thread(self._thread.join, 60)

    # ------------------------------------------------------------------ request API (loop)
    def submit(
        self,
        prompt_ids: list[int],
        params: SamplingParams,
        timeout_s: float,
        tenant: str = "default",
    ) -> RequestHandle:
        if self._closing:
            raise EngineClosed("engine is shutting down")
        cfg = self.config
        # In flight = running batch slots + bounded wait queue. Counting in-flight (rather than
        # "not yet started") means max_waiting=0 still accepts work while batch slots are free.
        if len(self._active) >= cfg.max_num_seqs + cfg.max_waiting:
            raise EngineOverloaded(f"server queue full ({len(self._queued)} waiting)")
        if self._tenant_queued[tenant] >= cfg.max_waiting_per_tenant:
            raise EngineOverloaded(f"too many queued requests for tenant {tenant!r}", True)
        if self._queued_tokens + len(prompt_ids) > cfg.max_queued_tokens:
            raise EngineOverloaded("queued prompt-token budget exhausted")
        handle = RequestHandle(
            prompt_ids, params, timeout_s, tenant, self._on_finish, self._wake_event.set
        )
        seq = Sequence(
            handle.id,
            prompt_ids,
            params,
            tenant,
            handle.timings.arrival,
            handle.deadline,
            handle=handle,
            cache_salt=tenant.encode() if cfg.prefix_cache_scope == "tenant" else b"",
        )
        self._active[handle.id] = handle
        self._queued.add(handle.id)
        self._tenant_queued[tenant] += 1
        self._queued_tokens += len(prompt_ids)
        with self._inbox_lock:
            self._inbox.append(seq)
        self._wake_event.set()
        return handle

    def speculation_stats(self) -> dict[str, Any] | None:
        if self.proposer is None:
            return None
        drafted, accepted = self.spec_draft_tokens, self.spec_accepted_tokens
        return {
            "method": self.proposer.method,
            "num_speculative_tokens": self.proposer.num_tokens,
            "max_batch": self.config.spec_max_batch,
            "steps": self.spec_steps,
            "draft_tokens": drafted,
            "accepted_tokens": accepted,
            "acceptance_rate": round(accepted / drafted, 4) if drafted else None,
        }

    def get(self, request_id: str) -> RequestHandle | None:
        return self._active.get(request_id)

    @property
    def num_active(self) -> int:
        return len(self._active)

    @property
    def num_queued(self) -> int:
        return len(self._queued)

    def _unqueue(self, handle: RequestHandle) -> None:
        if handle.id in self._queued:
            self._queued.discard(handle.id)
            self._tenant_queued[handle.tenant] -= 1
            if not self._tenant_queued[handle.tenant]:
                del self._tenant_queued[handle.tenant]
            self._queued_tokens -= len(handle.prompt_ids)

    def _on_finish(self, handle: RequestHandle) -> None:
        self._active.pop(handle.id, None)
        self._unqueue(handle)
        f = handle.finish
        if self.metrics is not None:
            self.metrics.observe_request(handle)
        log.info(
            "request finished",
            extra={
                "request_id": handle.id,
                "tenant": handle.tenant,
                "reason": str(f.reason),
                "prompt_tokens": f.prompt_tokens,
                "completion_tokens": f.completion_tokens,
                "cached_tokens": f.cached_tokens,
                "preemptions": f.preemptions,
                **({"draft_tokens": f.draft_tokens} if f.draft_tokens else {}),
                **({"accepted_tokens": f.accepted_tokens} if f.draft_tokens else {}),
                **{k: v for k, v in f.timings.items()},
                **({"error": f.error} if f.error else {}),
            },
        )

    def _deliver(self, events: list[tuple[RequestHandle, Any]]) -> None:
        for handle, ev in events:
            if isinstance(ev, _Started):
                if handle.finish is None and handle.state == "queued":
                    handle.state = "running"
                    handle.timings.started = ev.t
                    self._unqueue(handle)
            else:
                handle._push(ev)

    # ------------------------------------------------------------------ worker thread
    def _worker_main(self) -> None:
        while not self._stopping:
            self._wake_event.clear()
            events: list[tuple[RequestHandle, Any]] = []
            try:
                self._drain_inbox(events)
                self._reap(events)
                if self.scheduler.has_work():
                    out = self.scheduler.schedule()
                    if out:
                        self._run_step(out, events)
                    else:
                        self._wake_event.wait(0.002)
                else:
                    self._wake_event.wait(0.25)
            except Exception as e:  # a bug in the loop must not strand every request
                log.exception("engine step failed; failing running requests")
                self.fatal_error = f"{type(e).__name__}: {e}"
                for seq in list(self.scheduler.running):
                    self._finish(seq, FinishReason.ERROR, events, error=self.fatal_error)
            if events:
                self._post(events)
        # shutdown: release whatever is left (handles were already cancelled by stop())
        events = []
        self._drain_inbox(events)
        for seq in list(self.scheduler.running) + list(self.scheduler.waiting):
            self._finish(seq, FinishReason.CANCELLED, events)
        if events:
            self._post(events)

    def _post(self, events: list[tuple[RequestHandle, Any]]) -> None:
        try:
            self._loop.call_soon_threadsafe(self._deliver, events)
        except RuntimeError:  # loop already closed during interpreter shutdown
            pass

    def _drain_inbox(self, events) -> None:
        with self._inbox_lock:
            new, self._inbox = self._inbox, []
        tokenizer = self.backend.tokenizer
        for seq in new:
            seq.detok = IncrementalDetokenizer(tokenizer, seq.prompt_ids)
            seq.stopper = StopChecker(seq.params.stop) if seq.params.stop else None
            self.scheduler.add(seq)

    def _interrupted(self, seq: Sequence, now: float) -> FinishReason | None:
        h = seq.handle
        if h is not None and h._cancel_flag.is_set():
            return h._cancel_reason
        if now >= seq.deadline:
            return FinishReason.TIMEOUT
        return None

    def _reap(self, events) -> None:
        now = time.monotonic()
        for seq in list(self.scheduler.running) + list(self.scheduler.waiting):
            reason = self._interrupted(seq, now)
            if reason is not None:
                self._finish(seq, reason, events)

    def _run_step(self, out: SchedulerOutput, events) -> None:
        now = time.monotonic()
        started = []
        for seq in out.admitted:
            if seq.first_scheduled is None:
                seq.first_scheduled = now
                started.append((seq.handle, _Started(now)))
        if started:
            # Post *before* the (possibly long) step, so the loop sees "running" immediately —
            # otherwise a cancel during the first prefill would be treated as a queued cancel.
            events.extend(started)
            self._post(list(events))
            events.clear()
        t0 = time.perf_counter()
        try:
            tokens = self.backend.execute(out.scheduled)
        except Exception as e:
            log.exception("backend step failed")
            err = f"{type(e).__name__}: {e}"
            for s in out.scheduled:
                if s.seq.status != SeqStatus.FINISHED:
                    self._finish(s.seq, FinishReason.ERROR, events, error=err)
            return
        step_s = time.perf_counter() - t0
        self.num_steps += 1
        if out.num_draft_tokens:
            self.spec_steps += 1
            self.spec_draft_tokens += out.num_draft_tokens
            self.spec_accepted_tokens += sum(s.accepted for s in out.scheduled)
        self.scheduler.complete_step(out)
        now = time.monotonic()
        for s, tok in zip(out.scheduled, tokens, strict=True):
            seq = s.seq
            if tok is None or seq.status == SeqStatus.FINISHED:
                continue
            seq.num_draft_tokens += len(s.draft)
            seq.num_accepted_tokens += s.accepted
            reason = self._interrupted(seq, now)
            if reason is not None:  # computed after cancel/deadline: discard the token(s)
                self._finish(seq, reason, events)
                continue
            for t in (*s.draft[: s.accepted], tok):
                if not self._append_token(seq, t, now, events):
                    break
            else:
                self.scheduler.accept_draft(seq, s.accepted)
        if self.metrics is not None:
            self.metrics.observe_step(self, out, step_s)

    def _append_token(self, seq: Sequence, tok: int, now: float, events) -> bool:
        """Append one generated token and emit it; False if the sequence finished."""
        if tok in self.backend.tokenizer.eos_token_ids and not seq.params.ignore_eos:
            self._finish(seq, FinishReason.STOP, events)
            return False
        seq.token_ids.append(tok)
        text = seq.detok.add(tok)
        stopped = False
        if seq.stopper is not None:
            text, stopped = seq.stopper.process(text)
        events.append((seq.handle, TokenEvent(seq.num_output_tokens - 1, tok, text, now)))
        if stopped:
            self._finish(seq, FinishReason.STOP, events, flush=False)
            return False
        if seq.num_output_tokens >= seq.params.max_tokens:
            self._finish(seq, FinishReason.LENGTH, events)
            return False
        return True

    def _finish(
        self,
        seq: Sequence,
        reason: FinishReason,
        events,
        error: str | None = None,
        flush: bool = True,
    ) -> None:
        tail = ""
        if flush and reason != FinishReason.ERROR and seq.detok is not None:
            tail = seq.detok.flush()
            if seq.stopper is not None:
                tail, stopped = seq.stopper.process(tail)
                if stopped:
                    reason = FinishReason.STOP
                else:
                    tail += seq.stopper.flush()
        self.scheduler.finish(seq)
        events.append(
            (
                seq.handle,
                FinishEvent(
                    reason,
                    seq.num_prompt_tokens,
                    text=tail,
                    error=error,
                    cached_tokens=seq.num_cached_tokens,
                    preemptions=seq.num_preemptions,
                    draft_tokens=seq.num_draft_tokens,
                    accepted_tokens=seq.num_accepted_tokens,
                ),
            )
        )
