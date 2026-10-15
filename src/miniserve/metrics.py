"""Prometheus metrics (``GET /metrics``).

Names follow vLLM's where an equivalent exists, so existing Grafana panels translate. Each app
gets its own ``CollectorRegistry`` (no global state: tests can build many apps per process).
Step-level metrics are observed on the engine worker thread, request-level ones on the event
loop thread; ``prometheus_client`` is thread-safe.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import psutil
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

if TYPE_CHECKING:
    from miniserve.engine import Engine, RequestHandle
    from miniserve.scheduler import SchedulerOutput

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80)
TOKEN_LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3, 0.6, 1.2, 2.5)


class Metrics:
    def __init__(self) -> None:
        r = self.registry = CollectorRegistry()
        self.requests = Counter(
            "miniserve_requests", "Finished requests", ["finish_reason"], registry=r
        )
        self.rejected = Counter(
            "miniserve_requests_rejected",
            "Requests rejected before queueing",
            ["reason"],
            registry=r,
        )
        self.ttft = Histogram(
            "miniserve_time_to_first_token_seconds",
            "Arrival to first generated token",
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.e2e = Histogram(
            "miniserve_e2e_request_latency_seconds",
            "Arrival to finish",
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.queue = Histogram(
            "miniserve_request_queue_time_seconds",
            "Arrival to first scheduled step",
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.tpot = Histogram(
            "miniserve_time_per_output_token_seconds",
            "Mean decode time per token, per request",
            buckets=TOKEN_LATENCY_BUCKETS,
            registry=r,
        )
        self.prompt_tokens = Counter(
            "miniserve_prompt_tokens", "Prompt tokens of finished requests", registry=r
        )
        self.cached_tokens = Counter(
            "miniserve_prefix_cache_hit_tokens",
            "Prompt tokens served from the prefix cache",
            registry=r,
        )
        self.generation_tokens = Counter(
            "miniserve_generation_tokens", "Generated tokens", registry=r
        )
        self.preemptions = Counter(
            "miniserve_preemptions", "Sequences preempted (recompute)", registry=r
        )
        self.step_time = Histogram(
            "miniserve_step_duration_seconds",
            "Model step wall time",
            buckets=TOKEN_LATENCY_BUCKETS,
            registry=r,
        )
        self.batch_seqs = Histogram(
            "miniserve_batch_sequences",
            "Sequences per step",
            buckets=(1, 2, 4, 8, 16, 32, 64, 128),
            registry=r,
        )
        self.batch_tokens = Histogram(
            "miniserve_batch_tokens",
            "Tokens per step",
            buckets=(1, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096),
            registry=r,
        )
        self.running = Gauge(
            "miniserve_num_requests_running", "Sequences in the running batch", registry=r
        )
        self.waiting = Gauge(
            "miniserve_num_requests_waiting", "Sequences waiting for admission", registry=r
        )
        self.kv_usage = Gauge(
            "miniserve_kv_cache_usage_ratio", "Fraction of KV blocks in use", registry=r
        )
        self.kv_blocks = Gauge("miniserve_kv_cache_blocks", "KV blocks", ["state"], registry=r)
        self.prefix_hit_rate = Gauge(
            "miniserve_prefix_cache_hit_rate", "Cumulative hit tokens / queried tokens", registry=r
        )
        self.rss = Gauge("miniserve_process_resident_memory_bytes", "Process RSS", registry=r)
        self.device_mem = Gauge(
            "miniserve_device_memory_bytes", "Accelerator / KV memory", ["kind"], registry=r
        )
        self._proc = psutil.Process()

    # worker thread
    def observe_step(self, engine: Engine, out: SchedulerOutput, step_s: float) -> None:
        self.step_time.observe(step_s)
        self.batch_seqs.observe(len(out.scheduled))
        self.batch_tokens.observe(out.num_tokens)
        if out.preempted:
            self.preemptions.inc(len(out.preempted))
        self._set_engine_gauges(engine)

    def _set_engine_gauges(self, engine: Engine) -> None:
        bm = engine.block_manager
        self.running.set(engine.scheduler.num_running)
        self.waiting.set(engine.scheduler.num_waiting)
        self.kv_usage.set(bm.num_used_blocks / bm.num_blocks)

    # loop thread
    def observe_request(self, handle: RequestHandle) -> None:
        f, t = handle.finish, handle.timings
        self.requests.labels(str(f.reason)).inc()
        self.prompt_tokens.inc(f.prompt_tokens)
        self.cached_tokens.inc(f.cached_tokens)
        self.generation_tokens.inc(f.completion_tokens)
        if t.first_token is not None:
            self.ttft.observe(t.first_token - t.arrival)
        if t.started is not None:
            self.queue.observe(t.started - t.arrival)
        if t.finished is not None:
            self.e2e.observe(t.finished - t.arrival)
        if f.completion_tokens >= 2 and t.last_token > t.first_token:
            self.tpot.observe((t.last_token - t.first_token) / (f.completion_tokens - 1))

    def render(self, engine: Engine | None) -> bytes:
        self.rss.set(self._proc.memory_info().rss)
        if engine is not None:
            self._set_engine_gauges(engine)
            bm = engine.block_manager
            self.kv_blocks.labels("used").set(bm.num_used_blocks)
            self.kv_blocks.labels("cached_free").set(len(bm._evictable))
            self.kv_blocks.labels("total").set(bm.num_blocks)
            q = bm.stats.queried_tokens
            self.prefix_hit_rate.set(bm.stats.hit_tokens / q if q else 0.0)
            for kind, value in engine.backend.memory_stats().items():
                self.device_mem.labels(kind.removesuffix("_bytes")).set(value)
        return generate_latest(self.registry)
