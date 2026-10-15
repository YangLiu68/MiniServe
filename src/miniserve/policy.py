"""Waiting-queue policies: which request is admitted into the running batch next.

* ``fcfs`` — global arrival order.
* ``fair`` — Virtual Token Counter (Sheng et al., "Fairness in Serving Large Language Models",
  OSDI 2024): each tenant has a counter of weighted tokens it has been served; the next request
  comes from the backlogged tenant with the smallest counter. When a tenant becomes backlogged
  its counter is lifted to the minimum of the currently backlogged tenants — or, if nobody is
  waiting, to the counter of the tenant whose request last left the queue (it is the one being
  served) — so idle time cannot be banked as credit. Input tokens weigh 1, output tokens 2 (the
  paper's default weights).

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
    # Idle tenants' counters are forgotten beyond this many; only counters at or below the
    # lift floor are dropped, which a returning tenant would be lifted to anyway.
    max_idle_tenants = 10_000

    def __init__(self, input_weight: float = 1.0, output_weight: float = 2.0) -> None:
        self.w_in, self.w_out = input_weight, output_weight
        self._queues: dict[str, deque[Sequence]] = {}  # backlogged tenants only
        self._len = 0
        self._last_left: str | None = None  # tenant whose request was admitted most recently
        self.counters: dict[str, float] = {}

    def _floor(self) -> float:
        if self._queues:
            return min(self.counters[t] for t in self._queues)
        if self._last_left is not None:
            return self.counters.get(self._last_left, 0.0)
        return 0.0

    def push(self, seq: Sequence, front: bool = False) -> None:
        q = self._queues.get(seq.tenant)
        if q is None:  # tenant becomes backlogged: lift its counter (no banking of idle credit)
            floor = self._floor()
            self.counters[seq.tenant] = max(self.counters.get(seq.tenant, 0.0), floor)
            q = self._queues[seq.tenant] = deque()
            if len(self.counters) > self.max_idle_tenants + len(self._queues):
                self._forget_idle(floor)
        (q.appendleft if front else q.append)(seq)
        self._len += 1

    def _forget_idle(self, floor: float) -> None:
        for tenant in [t for t, c in self.counters.items() if c <= floor]:
            if tenant not in self._queues and tenant != self._last_left:
                del self.counters[tenant]

    def peek(self) -> Sequence | None:
        best = None
        for tenant, q in self._queues.items():
            key = (self.counters[tenant], q[0].arrival)
            if best is None or key < best[0]:
                best = (key, q[0])
        return best[1] if best else None

    def _dequeued(self, seq: Sequence, q: deque[Sequence]) -> None:
        self._len -= 1
        if not q:
            del self._queues[seq.tenant]

    def pop(self, seq: Sequence) -> None:
        q = self._queues[seq.tenant]
        assert q[0] is seq
        q.popleft()
        self._last_left = seq.tenant
        self._dequeued(seq, q)

    def remove(self, seq: Sequence) -> bool:
        q = self._queues.get(seq.tenant)
        if q is None:
            return False
        try:
            q.remove(seq)
        except ValueError:
            return False
        self._dequeued(seq, q)
        return True

    def charge(self, scheduled: list[ScheduledSeq]) -> None:
        """Bill tenants for tokens processed this step (recompute after preemption is free).

        Speculative draft tokens are billed as output tokens whether or not they are accepted:
        the tenant used that compute.
        """
        for s in scheduled:
            seq = s.seq
            end = s.start + s.num_query
            new_lo = max(s.start, seq.max_computed)
            if end <= new_lo:
                continue
            n_in = max(0, min(end, seq.num_prompt_tokens) - new_lo)
            n_out = (end - new_lo) - n_in
            self.counters[seq.tenant] = (
                self.counters.get(seq.tenant, 0.0) + self.w_in * n_in + self.w_out * n_out
            )

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[Sequence]:
        return iter([s for q in self._queues.values() for s in q])


def make_policy(name: str) -> FCFSPolicy | FairPolicy:
    if name == "fcfs":
        return FCFSPolicy()
    if name == "fair":
        return FairPolicy()
    raise ValueError(f"unknown scheduling policy {name!r}")
