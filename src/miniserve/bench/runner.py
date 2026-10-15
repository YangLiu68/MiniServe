"""Run a scenario against a live server: drive the streams, sample server metrics, summarize."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Any

import httpx

from miniserve.bench.client import RequestResult, send
from miniserve.bench.stats import summarize_by_tenant
from miniserve.bench.workload import PromptFactory, RequestSpec, Scenario, Stream, build_requests

log = logging.getLogger(__name__)

_METRIC_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)$")


def parse_prometheus(text: str) -> dict[str, float]:
    out = {}
    for line in text.splitlines():
        m = _METRIC_LINE.match(line.strip())
        if m:
            out[m.group(1) + (m.group(2) or "")] = float(m.group(3))
    return out


class MetricsSampler:
    """Polls MiniServe's /metrics during a run: peaks of memory/queue gauges + counter deltas."""

    GAUGES = (
        "miniserve_process_resident_memory_bytes",
        "miniserve_kv_cache_usage_ratio",
        "miniserve_num_requests_running",
        "miniserve_num_requests_waiting",
        'miniserve_device_memory_bytes{kind="device_allocated"}',
    )
    COUNTERS = (
        "miniserve_preemptions_total",
        "miniserve_prefix_cache_hit_tokens_total",
        "miniserve_batch_sequences_sum",
        "miniserve_batch_sequences_count",
        "miniserve_batch_tokens_sum",
        "miniserve_step_duration_seconds_sum",
    )

    def __init__(self, client: httpx.AsyncClient, interval: float = 0.5) -> None:
        self.client, self.interval = client, interval
        self.first: dict[str, float] = {}
        self.last: dict[str, float] = {}
        self.peaks: dict[str, float] = {}
        self.series: list[dict[str, float]] = []
        self._task: asyncio.Task | None = None
        self._t0 = 0.0

    async def _scrape(self) -> dict[str, float] | None:
        try:
            r = await self.client.get("/metrics", timeout=5)
            return parse_prometheus(r.text) if r.status_code == 200 else None
        except httpx.HTTPError:
            return None

    async def _loop(self) -> None:
        while True:
            m = await self._scrape()
            if m:
                self.last = m
                for g in self.GAUGES:
                    if g in m:
                        self.peaks[g] = max(self.peaks.get(g, 0.0), m[g])
                self.series.append(
                    {
                        "t": round(time.perf_counter() - self._t0, 2),
                        "running": m.get("miniserve_num_requests_running", 0.0),
                        "waiting": m.get("miniserve_num_requests_waiting", 0.0),
                        "kv": m.get("miniserve_kv_cache_usage_ratio", 0.0),
                    }
                )
            await asyncio.sleep(self.interval)

    async def start(self) -> bool:
        self._t0 = time.perf_counter()
        m = await self._scrape()
        if m is None or "miniserve_kv_cache_usage_ratio" not in m:
            return False  # not a MiniServe server
        self.first = m
        self._task = asyncio.create_task(self._loop())
        return True

    async def stop(self) -> dict[str, Any]:
        if self._task is None:
            return {}
        self._task.cancel()
        final = await self._scrape() or self.last
        delta = {c: final.get(c, 0.0) - self.first.get(c, 0.0) for c in self.COUNTERS}
        steps = delta["miniserve_batch_sequences_count"]
        return {
            "peak_rss_mb": round(self.peaks.get(self.GAUGES[0], 0) / 2**20, 1),
            "peak_kv_usage": round(self.peaks.get(self.GAUGES[1], 0), 4),
            "peak_running": self.peaks.get(self.GAUGES[2]),
            "peak_waiting": self.peaks.get(self.GAUGES[3]),
            "peak_device_mb": round(self.peaks.get(self.GAUGES[4], 0) / 2**20, 1) or None,
            "preemptions": delta["miniserve_preemptions_total"],
            "prefix_hit_tokens": delta["miniserve_prefix_cache_hit_tokens_total"],
            "steps": steps,
            "mean_batch_seqs": round(delta["miniserve_batch_sequences_sum"] / steps, 2)
            if steps
            else None,
            "mean_batch_tokens": round(delta["miniserve_batch_tokens_sum"] / steps, 1)
            if steps
            else None,
            "mean_step_ms": round(1000 * delta["miniserve_step_duration_seconds_sum"] / steps, 2)
            if steps
            else None,
            "series": self.series,
        }


async def _closed_loop(
    client,
    stream: Stream,
    specs: list[RequestSpec],
    t0: float,
    model,
    kind,
    results: list[RequestResult],
) -> None:
    it = iter(specs)
    deadline = t0 + stream.duration_s if stream.duration_s else None

    async def worker() -> None:
        while True:
            if deadline is not None and time.perf_counter() >= deadline:
                return
            try:
                spec = next(it)
            except StopIteration:
                return
            results.append(await send(client, spec, t0, model, kind))

    await asyncio.gather(*(worker() for _ in range(stream.concurrency)))


async def _open_loop(
    client,
    stream: Stream,
    specs: list[RequestSpec],
    t0: float,
    model,
    kind,
    results: list[RequestResult],
) -> None:
    tasks = []
    for spec in specs:
        if stream.duration_s is not None and spec.arrival > stream.duration_s:
            break
        delay = t0 + spec.arrival - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)

        async def one(s=spec) -> None:
            results.append(await send(client, s, t0, model, kind))

        tasks.append(asyncio.create_task(one()))
    await asyncio.gather(*tasks)


async def run_scenario(
    base_url: str,
    scenario: Scenario,
    factory: PromptFactory,
    model: str | None = None,
    server_kind: str = "miniserve",
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    specs = build_requests(scenario, factory)
    limits = httpx.Limits(max_connections=1000, max_keepalive_connections=200)
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s, limits=limits) as client:
        sampler = MetricsSampler(client)
        has_metrics = await sampler.start()
        results: list[RequestResult] = []
        load_before = os.getloadavg()
        t0 = time.perf_counter()
        drivers = [
            (_closed_loop if st.mode == "closed" else _open_loop)(
                client, st, specs[i], t0, model, server_kind, results
            )
            for i, st in enumerate(scenario.streams)
        ]
        await asyncio.gather(*drivers)
        wall = time.perf_counter() - t0
        server = await sampler.stop() if has_metrics else {}
    raw = [r.to_json() for r in results]
    warmup = max((s.warmup_s for s in scenario.streams), default=0.0)
    return {
        "scenario": scenario.name,
        "group": scenario.group,
        "x": scenario.x,
        "wall_s": round(wall, 3),
        "loadavg_before": [round(x, 2) for x in load_before],
        "loadavg_after": [round(x, 2) for x in os.getloadavg()],
        "summary": summarize_by_tenant(raw, warmup),
        "server": server,
        "results": raw,
    }
