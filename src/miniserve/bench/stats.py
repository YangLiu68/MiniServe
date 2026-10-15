"""Latency/throughput summaries for a set of request results."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (same convention as numpy's default)."""
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _dist(values: list[float], scale: float = 1000.0) -> dict[str, float | None]:
    def r(v):
        return None if v is None else round(v * scale, 2)

    return {
        "mean": r(sum(values) / len(values)) if values else None,
        "p50": r(percentile(values, 50)),
        "p90": r(percentile(values, 90)),
        "p95": r(percentile(values, 95)),
        "p99": r(percentile(values, 99)),
        "max": r(max(values)) if values else None,
    }


def summarize(results: Iterable[dict[str, Any]], warmup_s: float = 0.0) -> dict[str, Any]:
    """``results`` are ``RequestResult.to_json()`` dicts. Latencies are reported in ms."""
    rs = [r for r in results if r["start"] >= warmup_s]
    ok = [r for r in rs if r["error"] is None and r["finish_reason"] in ("stop", "length")]
    if not rs:
        return {"requests": 0}
    t_start = min(r["start"] for r in rs)
    t_end = max(r["start"] + (r["e2e"] or 0) for r in rs)
    duration = max(t_end - t_start, 1e-9)
    out_tokens = sum(r["output_tokens"] for r in ok)
    in_tokens = sum(r["prompt_tokens"] for r in ok)
    tpot = [
        (r["e2e"] - r["ttft"]) / (r["output_tokens"] - 1)
        for r in ok
        if r["ttft"] is not None and r["output_tokens"] > 1
    ]
    itl = [g for r in ok for g in r["itl"]]
    cached = sum(r.get("cached_tokens") or 0 for r in ok)
    return {
        "requests": len(rs),
        "completed": len(ok),
        "errors": len(rs) - len(ok),
        "error_rate": round((len(rs) - len(ok)) / len(rs), 4),
        "duration_s": round(duration, 3),
        "request_throughput": round(len(ok) / duration, 3),
        "output_token_throughput": round(out_tokens / duration, 2),
        "total_token_throughput": round((out_tokens + in_tokens) / duration, 2),
        "mean_prompt_tokens": round(in_tokens / len(ok), 1) if ok else None,
        "mean_output_tokens": round(out_tokens / len(ok), 1) if ok else None,
        "prefix_cached_fraction": round(cached / in_tokens, 4) if in_tokens else None,
        "ttft_ms": _dist([r["ttft"] for r in ok if r["ttft"] is not None]),
        "tpot_ms": _dist(tpot),
        "itl_ms": _dist(itl),
        "e2e_ms": _dist([r["e2e"] for r in ok]),
        "error_samples": sorted({r["error"] for r in rs if r["error"]})[:3],
    }


def summarize_by_tenant(results: list[dict[str, Any]], warmup_s: float = 0.0) -> dict[str, Any]:
    tenants = sorted({r["tenant"] for r in results})
    out = {"all": summarize(results, warmup_s)}
    if len(tenants) > 1:
        for t in tenants:
            out[t] = summarize([r for r in results if r["tenant"] == t], warmup_s)
    return out
