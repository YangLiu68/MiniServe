"""Waiting-queue policies: which request is admitted into the running batch next.

* ``fcfs`` — global arrival order.
* ``fair`` — Virtual Token Counter (Sheng et al., "Fairness in Serving Large Language Models",
  OSDI 2024): each tenant has a counter of weighted tokens it has been served; the next request
  comes from the backlogged tenant with the smallest counter. When a tenant becomes backlogged
  its counter is lifted to the minimum of the currently backlogged tenants, so idle time cannot
  be banked as credit. Input tokens weigh 1, output tokens 2 (the paper's default weights).

Preempted sequences go back to the *front* of their queue (they were admitted before anything
still waiting).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator

from miniserve.sequence import ScheduledSeq, Sequence


class FCFSPolicy:
    name = "fcfs"

    def __init__(self) -> None:
        self._q: deque[Sequence] = deque()

    def push(self, seq: Sequence, front: bool = False) -> None:
        (self._q.appendleft if front else self._q.append)(seq)

    def peek(self) -> Sequence | None:
        return self._q[0] if self._q else None

    def pop(self, seq: Sequence) -> None:
        assert self._q[0] is seq
        self._q.popleft()

    def remove(self, seq: Sequence) -> bool:
        try:
            self._q.remove(seq)
            return True
        except ValueError:
            return False

    def charge(self, scheduled: list[ScheduledSeq]) -> None:
        pass

    def __len__(self) -> int:
        return len(self._q)

    def __iter__(self) -> Iterator[Sequence]:
        return iter(list(self._q))


class FairPolicy:
    name = "fair"

    def __init__(self, input_weight: float = 1.0, output_weight: float = 2.0) -> None:
        self.w_in, self.w_out = input_weight, output_weight
        self._queues: dict[str, deque[Sequence]] = {}
        self.counters: dict[str, float] = {}

    def push(self, seq: Sequence, front: bool = False) -> None:
        q = self._queues.get(seq.tenant)
        if not q:  # tenant becomes backlogged: lift its counter (no banking of idle credit)
            backlogged = [self.counters[t] for t, tq in self._queues.items() if tq]
            floor = min(backlogged) if backlogged else 0.0
            self.counters[seq.tenant] = max(self.counters.get(seq.tenant, 0.0), floor)
            q = self._queues.setdefault(seq.tenant, deque())
        (q.appendleft if front else q.append)(seq)

    def peek(self) -> Sequence | None:
        best = None
        for tenant, q in self._queues.items():
            if not q:
                continue
            key = (self.counters[tenant], q[0].arrival)
            if best is None or key < best[0]:
                best = (key, q[0])
        return best[1] if best else None

    def pop(self, seq: Sequence) -> None:
        q = self._queues[seq.tenant]
        assert q[0] is seq
        q.popleft()

    def remove(self, seq: Sequence) -> bool:
        q = self._queues.get(seq.tenant)
        if q is None:
            return False
        try:
            q.remove(seq)
            return True
        except ValueError:
            return False

    def charge(self, scheduled: list[ScheduledSeq]) -> None:
        """Bill tenants for tokens processed this step (recompute after preemption is free)."""
        for s in scheduled:
            seq = s.seq
            end = s.start + s.num_tokens
            new_lo = max(s.start, seq.max_computed)
            if end <= new_lo:
                continue
            n_in = max(0, min(end, seq.num_prompt_tokens) - new_lo)
            n_out = (end - new_lo) - n_in
            self.counters[seq.tenant] = (
                self.counters.get(seq.tenant, 0.0) + self.w_in * n_in + self.w_out * n_out
            )

    def __len__(self) -> int:
        return sum(len(q) for q in self._queues.values())

    def __iter__(self) -> Iterator[Sequence]:
        return iter([s for q in self._queues.values() for s in q])


def make_policy(name: str) -> FCFSPolicy | FairPolicy:
    if name == "fcfs":
        return FCFSPolicy()
    if name == "fair":
        return FairPolicy()
    raise ValueError(f"unknown scheduling policy {name!r}")
