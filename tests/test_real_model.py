"""Opt-in tests against the real default model (downloads ~270 MB on first run).

Run with: MINISERVE_RUN_REAL_MODEL_TESTS=1 pytest -m real_model
CPU + float32 so results are deterministic across machines.
"""

import pytest

pytest.importorskip("torch")

from miniserve.config import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION  # noqa: E402
from miniserve.detokenizer import IncrementalDetokenizer  # noqa: E402

from conftest import app_client, parse_sse, run_sync  # noqa: E402

pytestmark = pytest.mark.real_model


@pytest.fixture(scope="module")
def models():
    from miniserve.backends.hf import HFBackend, load_hf_model
    from miniserve.backends.native import NativeBackend

    hf_model, tok = load_hf_model(
        DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, "cpu", __import__("torch").float32
    )
    native = NativeBackend.from_hf_model(hf_model, tok, DEFAULT_MODEL_ID, kv_cache_mb=64)
    hf = HFBackend(hf_model, tok, DEFAULT_MODEL_ID, kv_cache_mb=64)
    return native, hf


PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, in a small village by the sea,",
    "List three primary colors:",
]


def test_native_engine_matches_transformers_greedy(models):
    native, hf = models
    ids = [native.tokenizer.encode(p) for p in PROMPTS]
    a, _ = run_sync(native, ids, max_tokens=24, max_num_batched_tokens=32)  # batched + chunked
    b, _ = run_sync(hf, ids, max_tokens=24, max_num_seqs=1)  # sequential DynamicCache
    for sa, sb in zip(a, b, strict=True):
        assert sa.output_ids == sb.output_ids


def test_greedy_completion_and_chat_are_sensible(models):
    native, _ = models
    with app_client(native) as c:
        r = c.post(
            "/v1/completions",
            json={"prompt": "The capital of France is", "max_tokens": 8, "temperature": 0},
        )
        assert "Paris" in r.json()["choices"][0]["text"]
        r = c.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "What is the capital of France?"}],
                "max_tokens": 24,
                "temperature": 0,
            },
        )
        assert "Paris" in r.json()["choices"][0]["message"]["content"]


def test_stream_equals_non_stream_and_prefix_cache_hits(models):
    native, _ = models
    system = "You are a concise assistant that answers in one sentence. " * 4
    with app_client(native) as c:
        body = {"prompt": system + "Why is the sky blue?", "max_tokens": 32, "temperature": 0}
        full = c.post("/v1/completions", json=body).json()
        chunks, done = parse_sse(c.post("/v1/completions", json={**body, "stream": True}).text)
        assert done
        assert "".join(ch["choices"][0]["text"] for ch in chunks) == full["choices"][0]["text"]
        cached = chunks[-1]["usage"]["prompt_tokens_details"]["cached_tokens"]
        assert cached >= 48  # second identical prompt reuses full blocks


def test_incremental_detokenizer_on_real_tokenizer(models):
    tok = models[0].tokenizer
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


def test_speculative_decoding_is_token_identical_on_the_real_model(models):
    """n-gram speculative decoding (batched, with draft verification) must reproduce the
    Transformers backend's greedy output exactly, and actually accept drafts on copy tasks."""
    from miniserve.speculative import NgramProposer

    native, hf = models
    tasks = [
        "Correct the spelling: The libary opens at nine oclock every morning, exept on Sundays.",
        "Rewrite the list as a numbered list:\n- red apples\n- green pears\n- yellow bananas\n",
        "def add(a, b):\n    return a + b\n\n\n"
        "def sub(a, b):\n    return a - b\n\n\ndef mul(a, b):",
    ]
    ids = [native.tokenizer.apply_chat_template([{"role": "user", "content": t}]) for t in tasks]
    a, _ = run_sync(native, ids, max_tokens=48, proposer=NgramProposer(4, 3, 2))
    b, _ = run_sync(hf, ids, max_tokens=48, max_num_seqs=1)
    for sa, sb in zip(a, b, strict=True):
        assert sa.output_ids == sb.output_ids
    assert sum(s.num_accepted_tokens for s in a) >= 10
