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
    """Wraps a backend and counts tokens it actually produced (i.e. forward passes spent)."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.name, self.model_id, self.device = inner.name, inner.model_id, inner.device
        self.max_context, self.tokenizer = inner.max_context, inner.tokenizer
        self.yielded = 0
        self.calls: list[list[int]] = []

    def generate(self, prompt_ids: list[int], params: SamplingParams):
        self.calls.append(prompt_ids)
        for tok in self.inner.generate(prompt_ids, params):
            self.yielded += 1
            yield tok

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
