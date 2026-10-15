"""Iteration-level (continuous) batching scheduler with chunked prefill and preemption.

Every engine step the scheduler builds one batch under a token budget
(``max_num_batched_tokens``) and a sequence limit (``max_num_seqs``):

1. **Decodes first.** Every running sequence that only needs its next token gets 1 token of
   budget. Serving decodes before prefills keeps inter-token latency steady when long prompts
   arrive (the "decode-maximal" batching of Sarathi-Serve, Agrawal et al. 2024).
2. **Ongoing prefills** of running sequences get what is left, in admission order. With chunked
   prefill a long prompt is split across steps instead of stalling everyone for one huge step.
3. **Admission.** Waiting sequences (chosen by the policy: FCFS or fair) are admitted while
   budget, sequence slots and KV blocks remain. A prefix-cache hit means only the uncached tail
   is scheduled. Admission stops at the first sequence that does not fit, so the policy's
   order is respected (no skipping ahead).

If a running sequence needs a new KV block and none is free, the *youngest* running sequence is
preempted: its blocks are freed and it goes back to the front of the waiting queue to be
recomputed later (vLLM's recompute preemption). Already-streamed tokens are kept; with prefix
caching, much of the recompute is usually a cache hit. Nothing is admitted in a step that
preempted, to avoid thrashing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from miniserve.block_manager import BlockManager
from miniserve.policy import FairPolicy, FCFSPolicy
from miniserve.sequence import ScheduledSeq, SeqStatus, Sequence


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 512
    enable_chunked_prefill: bool = True
    # Free blocks kept in reserve when admitting new sequences, so running sequences can grow
    # for a while before anyone has to be preempted.
    watermark_blocks: int = 0

    def __post_init__(self) -> None:
        if self.max_num_seqs < 1 or self.max_num_batched_tokens < 1:
            raise ValueError("max_num_seqs and max_num_batched_tokens must be >= 1")


@dataclass
class SchedulerOutput:
    scheduled: list[ScheduledSeq] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)
    admitted: list[Sequence] = field(default_factory=list)

    @property
    def num_tokens(self) -> int:
        return sum(s.num_tokens for s in self.scheduled)

    @property
    def num_prefill_tokens(self) -> int:
        return sum(s.num_tokens for s in self.scheduled if not _is_decode(s))

    def __bool__(self) -> bool:
        return bool(self.scheduled)


def _is_decode(s: ScheduledSeq) -> bool:
    return s.num_tokens == 1 and s.start >= s.seq.num_prompt_tokens


class Scheduler:
    def __init__(
        self,
        config: SchedulerConfig,
        block_manager: BlockManager,
        policy: FCFSPolicy | FairPolicy,
        on_release: Callable[[Sequence, bool], None] = lambda seq, preempted: None,
    ) -> None:
        self.config = config
        self.bm = block_manager
        self.waiting = policy
        self.running: list[Sequence] = []  # admission order: oldest first
        self._on_release = on_release  # backend hook: (seq, preempted) -> drop per-seq state
        self.num_preemptions = 0

    # ------------------------------------------------------------------ queue ops
    def add(self, seq: Sequence) -> None:
        seq.status = SeqStatus.WAITING
        self.waiting.push(seq)

    @property
    def num_waiting(self) -> int:
        return len(self.waiting)

    @property
    def num_running(self) -> int:
        return len(self.running)

    def has_work(self) -> bool:
        return bool(self.running) or len(self.waiting) > 0

    def finish(self, seq: Sequence) -> None:
        """Remove a sequence (finished, cancelled or failed) and release its resources."""
        if seq.status == SeqStatus.RUNNING:
            self.running.remove(seq)
        elif seq.status == SeqStatus.WAITING:
            self.waiting.remove(seq)
        if seq.block_ids:
            self.bm.free(seq)
        self._on_release(seq, False)
        seq.status = SeqStatus.FINISHED

    def _preempt(self, seq: Sequence) -> None:
        self.running.remove(seq)
        self.bm.free(seq)
        self._on_release(seq, True)
        seq.num_computed = 0
        seq.num_preemptions += 1
        seq.status = SeqStatus.WAITING
        self.waiting.push(seq, front=True)
        self.num_preemptions += 1

    # ------------------------------------------------------------------ the step
    def schedule(self) -> SchedulerOutput:
        cfg, out = self.config, SchedulerOutput()
        budget = cfg.max_num_batched_tokens
        planned: dict[int, ScheduledSeq] = {}  # id(seq) -> its slot in this step

        def plan_running(seq: Sequence, n: int) -> bool:
            """Reserve blocks for ``n`` more tokens of a running seq, preempting if needed."""
            nonlocal budget
            target = seq.num_computed + n
            while not self.bm.can_allocate(seq, target):
                victim = self.running[-1]
                if victim in out.preempted:
                    break  # pragma: no cover - running never contains preempted seqs
                if id(victim) in planned:  # refund a victim already planned this step
                    budget += planned.pop(id(victim)).num_tokens
                self._preempt(victim)
                out.preempted.append(victim)
                if victim is seq:
                    return False
            self.bm.allocate(seq, target)
            planned[id(seq)] = ScheduledSeq(seq, seq.num_computed, n)
            budget -= n
            return True

        # 1. decodes, 2. ongoing prefills (both in admission order)
        for want_decode in (True, False):
            for seq in list(self.running):
                if budget <= 0:
                    break
                if seq.status != SeqStatus.RUNNING or id(seq) in planned:
                    continue
                remaining = seq.num_tokens - seq.num_computed
                if (remaining == 1) != want_decode:
                    continue
                n = min(remaining, budget)
                if n < remaining and not cfg.enable_chunked_prefill:
                    continue
                plan_running(seq, n)

        # 3. admit waiting sequences (not in a step that had to preempt)
        while not out.preempted and budget > 0 and len(self.running) < cfg.max_num_seqs:
            seq = self.waiting.peek()
            if seq is None:
                break
            hit = self.bm.match_prefix(seq)
            remaining = seq.num_tokens - hit.num_tokens
            n = min(remaining, budget)
            if n < remaining and not cfg.enable_chunked_prefill:
                break
            # The reserve only applies when someone is running; otherwise a sequence that needs
            # the whole cache could never start (validated at submit to fit on its own).
            reserve = cfg.watermark_blocks if self.running else 0
            if not self.bm.can_admit(hit, hit.num_tokens + n, reserve=reserve):
                break
            self.waiting.pop(seq)
            self.bm.admit(seq, hit, hit.num_tokens + n)
            seq.num_computed = hit.num_tokens
            if seq.num_preemptions == 0:
                seq.num_cached_tokens = hit.num_tokens
            seq.status = SeqStatus.RUNNING
            self.running.append(seq)
            out.admitted.append(seq)
            planned[id(seq)] = ScheduledSeq(seq, seq.num_computed, n)
            budget -= n

        # keep batch order stable (admission order) for reproducible step composition
        order = {id(s): i for i, s in enumerate(self.running)}
        out.scheduled = sorted(planned.values(), key=lambda s: order[id(s.seq)])
        self.waiting.charge(out.scheduled)
        return out

    def complete_step(self, out: SchedulerOutput) -> None:
        """Advance ``num_computed`` after the model ran ``out`` and register full blocks."""
        for s in out.scheduled:
            seq = s.seq
            if seq.status != SeqStatus.RUNNING:
                continue
            seq.num_computed = s.start + s.num_tokens
            seq.max_computed = max(seq.max_computed, seq.num_computed)
            self.bm.commit(seq)
