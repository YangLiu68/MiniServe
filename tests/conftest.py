from __future__ import annotations

import json
import os
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import uvicorn
from fastapi.testclient import TestClient

from miniserve.backends.base import SamplingParams
from miniserve.backends.toy import ToyBackend
from miniserve.config import Settings
from miniserve.server import create_app


def pytest_collection_modifyitems(config, items):
    if os.environ.get("MINISERVE_RUN_REAL_MODEL_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="set MINISERVE_RUN_REAL_MODEL_TESTS=1 to run real-model tests")
    for item in items:
        if "real_model" in item.keywords:
            item.add_marker(skip)


def make_settings(**overrides) -> Settings:
    """Settings isolated from the developer's environment / .env file."""
    base = dict(backend="toy", warmup=False, rate_limit_rpm=0)
    base.update(overrides)
    saved = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith("MINISERVE_")}
    try:
        return Settings(_env_file=None, **base)
    finally:
        os.environ.update(saved)


class RecordingBackend:
    """Wraps a backend; records what actually reached the model (tokens produced, requests,
    batch sizes) so tests can assert on wasted compute, not just on responses."""

    def __init__(self, inner) -> None:
        self.inner = inner
        for attr in (
            "name",
            "model_id",
            "device",
            "dtype",
            "max_context",
            "tokenizer",
            "block_size",
            "num_kv_blocks",
            "supports_prefix_caching",
            "supports_speculation",
        ):
            setattr(self, attr, getattr(inner, attr))
        self.produced = 0  # sampled tokens
        self.steps = 0
        self.max_batch = 0
        self.seen: set[str] = set()  # request ids that reached the model

    def execute(self, batch):
        self.steps += 1
        self.max_batch = max(self.max_batch, len(batch))
        self.seen.update(s.seq.request_id for s in batch)
        out = self.inner.execute(batch)
        self.produced += sum(t is not None for t in out) + sum(s.accepted for s in batch)
        return out

    def release(self, seq, preempted=False):
        self.inner.release(seq, preempted)

    def memory_stats(self):
        return self.inner.memory_stats()

    def close(self) -> None:
        self.inner.close()


def wait_ready(client, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.get("/readyz").status_code == 200:
            return
        time.sleep(0.01)
    raise AssertionError(f"server not ready: {client.get('/readyz').json()}")


@contextmanager
def app_client(backend=None, **settings) -> Iterator[TestClient]:
    app = create_app(make_settings(**settings), backend=backend)
    with TestClient(app) as client:
        wait_ready(client)
        yield client


@pytest.fixture
def toy_client() -> Iterator[TestClient]:
    with app_client(ToyBackend()) as c:
        yield c


def parse_sse(text: str) -> tuple[list[dict], bool]:
    """Return (json chunks, saw_done) from an SSE body."""
    chunks, done = [], False
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        assert block.startswith("data: "), block
        payload = block[len("data: ") :]
        if payload == "[DONE]":
            done = True
        else:
            assert not done, "chunk after [DONE]"
            chunks.append(json.loads(payload))
    return chunks, done


@contextmanager
def live_server(backend=None, **settings) -> Iterator[tuple[str, object]]:
    """Real uvicorn on a real socket (needed for genuine client-disconnect tests)."""
    app = create_app(make_settings(**settings), backend=backend)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="on"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            assert time.monotonic() < deadline, "uvicorn did not start"
            time.sleep(0.01)
        yield f"http://127.0.0.1:{port}", app
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()


def run_sync(
    backend,
    prompts,
    max_tokens=8,
    num_blocks=None,
    prefix=True,
    policy=None,
    temperature=0.0,
    seed=None,
    proposer=None,
    **sched_cfg,
):
    """Drive scheduler + backend synchronously (no threads/asyncio); returns (seqs, scheduler).

    With a ``proposer`` (speculative decoding), verified draft tokens are appended too."""
    from miniserve.block_manager import BlockManager
    from miniserve.policy import FCFSPolicy
    from miniserve.scheduler import Scheduler, SchedulerConfig
    from miniserve.sequence import Sequence

    bm = BlockManager(
        num_blocks or backend.num_kv_blocks,
        backend.block_size,
        prefix and backend.supports_prefix_caching,
    )
    sched = Scheduler(
        SchedulerConfig(**sched_cfg),
        bm,
        policy if policy is not None else FCFSPolicy(),
        on_release=backend.release,
        proposer=proposer,
    )
    seqs = []
    for i, p in enumerate(prompts):
        params = SamplingParams(max_tokens, temperature=temperature, seed=seed, ignore_eos=True)
        seqs.append(Sequence(f"s{i}", list(p), params, "t", float(i), 1e18))
        sched.add(seqs[-1])
    steps = 0
    while sched.has_work():
        out = sched.schedule()
        toks = backend.execute(out.scheduled)
        sched.complete_step(out)
        for s, t in zip(out.scheduled, toks, strict=True):
            if t is None:
                continue
            s.seq.num_draft_tokens += len(s.draft)
            s.seq.num_accepted_tokens += s.accepted
            s.seq.token_ids += [*s.draft[: s.accepted], t]  # drafts never overshoot max_tokens
            assert s.seq.num_output_tokens <= max_tokens
            if s.seq.num_output_tokens >= max_tokens:
                sched.finish(s.seq)
            else:
                sched.accept_draft(s.seq, s.accepted)
        steps += 1
        assert steps < 100_000
        bm.check_invariants(sched.running)
    sched.num_steps = steps
    return seqs, sched
