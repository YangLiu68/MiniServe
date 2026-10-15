"""Real PyTorch/Transformers code path on a tiny randomly initialised GPT-2 (no downloads)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from miniserve.backends.base import SamplingParams  # noqa: E402
from miniserve.backends.hf import build_tiny_random, sample_next  # noqa: E402

from conftest import app_client, parse_sse  # noqa: E402

pytestmark = pytest.mark.torch


@pytest.fixture(scope="module")
def tiny():
    return build_tiny_random(device="cpu", seed=0)


def take(backend, prompt_ids, n, **kw):
    """First n tokens from the manual loop, stopping early at EOS like the engine does."""
    out = []
    it = backend.generate(prompt_ids, SamplingParams(max_tokens=n, **kw))
    for tok in it:
        out.append(tok)
        if len(out) == n or tok in backend.tokenizer.eos_token_ids:
            break
    it.close()
    return out


@pytest.mark.parametrize("text", ["Hello", "The quick brown fox", "é🙂", "a" * 100])
def test_manual_kv_cache_loop_matches_hf_generate_greedy(tiny, text):
    ids = tiny.tokenizer.encode(text)
    ours = take(tiny, ids, 24, temperature=0.0)
    with torch.inference_mode():
        ref = tiny.model.generate(
            torch.tensor([ids]),
            max_new_tokens=24,
            do_sample=False,
            eos_token_id=256,
            pad_token_id=256,
        )[0, len(ids) :].tolist()
    assert ours == ref[: len(ours)]
    assert len(ours) == len(ref)


def test_seeded_sampling_is_reproducible(tiny):
    ids = tiny.tokenizer.encode("seed test")
    a = take(tiny, ids, 32, temperature=1.0, seed=1234)
    b = take(tiny, ids, 32, temperature=1.0, seed=1234)
    c = take(tiny, ids, 32, temperature=1.0, seed=4321)
    assert a == b
    assert a != c


def test_tiny_top_p_collapses_to_greedy(tiny):
    ids = tiny.tokenizer.encode("nucleus")
    greedy = take(tiny, ids, 16, temperature=0.0)
    nucleus = take(tiny, ids, 16, temperature=1.0, top_p=1e-9, seed=7)
    assert nucleus == greedy


def test_sample_next_unit():
    logits = torch.tensor([0.0, 5.0, 1.0, 4.9])
    gen = torch.Generator().manual_seed(0)
    assert sample_next(logits, SamplingParams(1, temperature=0.0), gen) == 1
    # top_p=0.5 keeps only the top token here (p(top) ~ 0.5 > mass before it = 0)
    picks = {sample_next(logits, SamplingParams(1, 1.0, top_p=0.5), gen) for _ in range(50)}
    assert picks == {1}
    # top_p=0.9 keeps tokens 1 and 3 (mass of 1 alone < 0.9) and never 0
    picks = {sample_next(logits, SamplingParams(1, 1.0, top_p=0.9), gen) for _ in range(300)}
    assert picks == {1, 3}


def test_api_stream_equals_non_stream_on_tiny_model(tiny):
    with app_client(tiny) as c:
        body = {"prompt": "Hello there", "max_tokens": 20, "temperature": 0}
        full = c.post("/v1/completions", json=body).json()
        r = c.post("/v1/completions", json={**body, "stream": True})
        chunks, done = parse_sse(r.text)
        assert done
        streamed = "".join(ch["choices"][0]["text"] for ch in chunks)
        assert streamed == full["choices"][0]["text"]
        assert chunks[-1]["usage"] == full["usage"]
        seeded = {**body, "temperature": 0.8, "seed": 99}
        a = c.post("/v1/completions", json=seeded).json()["choices"][0]["text"]
        b = c.post("/v1/completions", json=seeded).json()["choices"][0]["text"]
        assert a == b
