"""Per-client token-bucket rate limiter with bounded memory (in-process, single replica)."""

from __future__ import annotations

import time
from collections import OrderedDict


class RateLimiter:
    def __init__(self, rate_per_min: float, burst: int, max_clients: int = 10_000) -> None:
        self.rate = rate_per_min / 60.0  # tokens per second
        self.burst = burst
        self.max_clients = max_clients
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()  # key -> (tokens, t)

    @property
    def enabled(self) -> bool:
        return self.rate > 0

    def acquire(self, key: str, now: float | None = None) -> float:
        """Take one token. Returns 0.0 if allowed, else seconds until a token is available."""
        if not self.enabled:
            return 0.0
        now = time.monotonic() if now is None else now
        tokens, last = self._buckets.pop(key, (float(self.burst), now))
        tokens = min(self.burst, tokens + (now - last) * self.rate)
        wait = 0.0
        if tokens >= 1.0:
            tokens -= 1.0
        else:
            wait = (1.0 - tokens) / self.rate
        self._buckets[key] = (tokens, now)  # re-insert as most recently used
        while len(self._buckets) > self.max_clients:
            self._buckets.popitem(last=False)
        return wait
