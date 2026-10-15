"""Generic Transformers backend (DynamicCache) on a non-Llama architecture: tiny random GPT-2."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from miniserve.backends.base import SamplingParams  # noqa: E402
from miniserve.backends.hf import HFBackend  # noqa: E402
from miniserve.backends.toy import ByteTokenizer  # noqa: E402
from miniserve.sampling import sample_next  # noqa: E402

from conftest import app_client, parse_sse, run_sync  # noqa: E402

pytestmark = pytest.mark.torch


@pytest.fixture(scope="module")
def gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    cfg = GPT2Config(
        vocab_size=257,
        n_positions=512,
        n_embd=64,
        n_layer=2,
        n_head=2,
        bos_token_id=256,
        eos_token_id=256,
    )
    model = GPT2LMHeadModel(cfg).eval()
    return HFBackend(model, ByteTokenizer(), "tiny-random-gpt2", kv_cache_mb=4, block_size=8), model


def reference_greedy(model, ids, n):
    ids = list(ids)
    out = []
    with torch.inference_mode():
        for _ in range(n):
            tok = int(model(torch.tensor([ids])).logits[0, -1].argmax())
            out.append(tok)
            ids.append(tok)
    return out


def test_interleaved_dynamic_cache_decoding_matches_reference(gpt2):
    backend, model = gpt2
    prompts = [list(b"Hello"), list(b"The quick brown fox"), list("é🙂".encode())]
    seqs, _ = run_sync(backend, prompts, max_tokens=16, max_num_batched_tokens=8)
    for s, p in zip(seqs, prompts, strict=True):
        assert s.output_ids == reference_greedy(model, p, 16)


def test_preemption_drops_and_rebuilds_dynamic_cache(gpt2):
    backend, model = gpt2
    prompts = [list(bytes([65 + i]) * 9) for i in range(3)]
    seqs, sched = run_sync(backend, prompts, max_tokens=12, num_blocks=7)
    assert sched.num_preemptions > 0
    for s, p in zip(seqs, prompts, strict=True):
        assert s.output_ids == reference_greedy(model, p, 12)
    assert not backend._caches  # every per-sequence cache was released


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


def test_api_stream_equals_non_stream_and_seeds_reproduce(gpt2):
    backend, _ = gpt2
    with app_client(backend) as c:
        body = {"prompt": "Hello there", "max_tokens": 20, "temperature": 0, "ignore_eos": True}
        full = c.post("/v1/completions", json=body).json()
        chunks, done = parse_sse(c.post("/v1/completions", json={**body, "stream": True}).text)
        assert done
        assert "".join(ch["choices"][0]["text"] for ch in chunks) == full["choices"][0]["text"]
        seeded = {**body, "temperature": 0.8, "seed": 99}
        a = c.post("/v1/completions", json=seeded).json()["choices"][0]["text"]
        b = c.post("/v1/completions", json=seeded).json()["choices"][0]["text"]
        assert a == b
