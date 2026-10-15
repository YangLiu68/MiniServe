"""Opt-in tests against the real default model (downloads ~270 MB on first run).

Run with: MINISERVE_RUN_REAL_MODEL_TESTS=1 pytest -m real_model
CPU + float32 for determinism across machines.
"""

import pytest

pytest.importorskip("torch")

from miniserve.config import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION  # noqa: E402
from miniserve.detokenizer import IncrementalDetokenizer  # noqa: E402

from conftest import app_client, parse_sse  # noqa: E402

pytestmark = pytest.mark.real_model


@pytest.fixture(scope="module")
def smollm():
    from miniserve.backends.hf import HFBackend

    backend = HFBackend.from_pretrained(DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, "cpu", "float32")
    yield backend
    backend.close()


def test_greedy_completion_is_sensible(smollm):
    with app_client(smollm) as c:
        body = {"prompt": "The capital of France is", "max_tokens": 8, "temperature": 0}
        r = c.post("/v1/completions", json=body)
        assert r.status_code == 200
        assert "Paris" in r.json()["choices"][0]["text"]


def test_stream_equals_non_stream(smollm):
    with app_client(smollm) as c:
        body = {"prompt": "Write a haiku about caches:", "max_tokens": 40, "temperature": 0}
        full = c.post("/v1/completions", json=body).json()
        chunks, done = parse_sse(c.post("/v1/completions", json={**body, "stream": True}).text)
        assert done
        assert "".join(ch["choices"][0]["text"] for ch in chunks) == full["choices"][0]["text"]


def test_incremental_detokenizer_on_real_tokenizer(smollm):
    tok = smollm.tokenizer
    for text in [
        "Hello world, this is a test.",
        "  leading spaces and\nnewlines\n\n  ",
        "Unicode: café, naïve, 世界, 🙂👩‍💻, Ω≈ç√",
        "def f(x):\n    return x**2  # code",
    ]:
        ids = tok.encode(text)
        d = IncrementalDetokenizer(tok)
        streamed = "".join(d.add(i) for i in ids) + d.flush()
        assert streamed == tok.decode(ids)
