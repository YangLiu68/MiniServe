"""Native paged-KV model vs the Transformers reference implementation (tiny random models)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from miniserve.backends.base import SamplingParams  # noqa: E402
from miniserve.backends.native import build_metadata, build_tiny_random  # noqa: E402
from miniserve.block_manager import BlockManager, PrefixHit  # noqa: E402
from miniserve.sequence import ScheduledSeq, Sequence  # noqa: E402

from conftest import run_sync  # noqa: E402

pytestmark = pytest.mark.torch
ATOL = 2e-5
ARCHS = ["llama", "qwen2", "mistral"]


@pytest.fixture(scope="module", params=ARCHS)
def tiny(request):
    return build_tiny_random(arch=request.param, block_size=4)


def prompt(n, base=7):
    return [(base * 31 + i * 17) % 256 for i in range(n)]


def hf_last_logits(hf_model, ids):
    with torch.inference_mode():
        return hf_model(torch.tensor([ids])).logits[0, -1].float()


def hf_greedy(hf_model, ids, n):
    """Reference greedy decode with full recompute every step (no cache involved)."""
    ids = list(ids)
    out = []
    for _ in range(n):
        tok = int(hf_last_logits(hf_model, ids).argmax())
        out.append(tok)
        ids.append(tok)
    return out


def native_logits(backend, entries):
    """entries: list of (seq, start, n); returns logits rows for sampling entries."""
    batch = [ScheduledSeq(s, st, n) for s, st, n in entries]
    meta = build_metadata(batch, backend.block_size, backend.device)
    return backend.model.forward(meta, backend.kv).float()


def new_seq(bm, ids, rid):
    s = Sequence(rid, list(ids), SamplingParams(4), "t", 0.0, 1e18)
    bm.admit(s, PrefixHit([], 0), len(ids))
    return s


def test_full_prefill_logits_match_hf(tiny):
    backend, hf = tiny
    bm = BlockManager(backend.num_kv_blocks, backend.block_size, False)
    for n in (1, 5, 13, 40):
        ids = prompt(n, base=n)
        s = new_seq(bm, ids, f"p{n}")
        got = native_logits(backend, [(s, 0, n)])[0]
        torch.testing.assert_close(got, hf_last_logits(hf, ids), atol=ATOL, rtol=0)
        bm.free(s)


def test_chunked_prefill_equals_full_prefill(tiny):
    backend, hf = tiny
    bm = BlockManager(backend.num_kv_blocks, backend.block_size, False)
    ids = prompt(23)
    for chunk in (1, 3, 7):
        s = new_seq(bm, ids, f"c{chunk}")
        start = 0
        while start < len(ids):
            n = min(chunk, len(ids) - start)
            rows = native_logits(backend, [(s, start, n)])
            start += n
        torch.testing.assert_close(rows[0], hf_last_logits(hf, ids), atol=ATOL, rtol=0)
        bm.free(s)


def test_mixed_ragged_batch_matches_individual_hf_runs(tiny):
    backend, hf = tiny
    bm = BlockManager(backend.num_kv_blocks, backend.block_size, False)
    a_ids, b_ids, c_ids = prompt(9, 1), prompt(17, 2), prompt(4, 3)
    a, b, c = (new_seq(bm, x, r) for x, r in ((a_ids, "a"), (b_ids, "b"), (c_ids, "c")))
    # a: already prefilled except its last token (decode-like 1-token span)
    native_logits(backend, [(a, 0, 8)])
    # b: first chunk done earlier; this step runs its middle chunk (no sample row)
    native_logits(backend, [(b, 0, 6)])
    rows = native_logits(backend, [(a, 8, 1), (b, 6, 11), (c, 0, 4)])
    for row, ids in zip(rows, (a_ids, b_ids, c_ids), strict=True):
        torch.testing.assert_close(row, hf_last_logits(hf, ids), atol=ATOL, rtol=0)


def test_greedy_generation_matches_reference(tiny):
    backend, hf = tiny
    prompts = [prompt(5, 1), prompt(18, 2), prompt(9, 3), prompt(31, 4)]
    seqs, _ = run_sync(backend, prompts, max_tokens=12, max_num_batched_tokens=16)
    for s, p in zip(seqs, prompts, strict=True):
        assert s.output_ids == hf_greedy(hf, p, 12)


def test_prefix_cache_hit_gives_identical_output(tiny):
    backend, hf = tiny
    shared = prompt(16, 9)  # 4 full blocks of 4
    prompts = [shared + [1, 2, 3], shared + [4, 5], shared + [1, 2, 3]]
    seqs, sched = run_sync(backend, prompts, max_tokens=6, max_num_seqs=1)  # one at a time
    assert sched.bm.stats.hit_tokens >= 32  # 2nd and 3rd request reuse the shared blocks
    assert seqs[2].num_cached_tokens == 16
    for s, p in zip(seqs, prompts, strict=True):
        assert s.output_ids == hf_greedy(hf, p, 6)


def test_preemption_under_memory_pressure_keeps_outputs_exact(tiny):
    backend, hf = tiny
    prompts = [prompt(10, k) for k in range(4)]
    # 12 blocks x 4 = 48 slots for four sequences that grow to 10 + 10 = 20 tokens each
    seqs, sched = run_sync(backend, prompts, max_tokens=10, num_blocks=12, prefix=False)
    assert sched.num_preemptions > 0
    for s, p in zip(seqs, prompts, strict=True):
        assert s.output_ids == hf_greedy(hf, p, 10)


def test_seeded_sampling_independent_of_batch_composition():
    backend, _ = build_tiny_random(arch="llama", block_size=4, seed=3)
    target = prompt(7, 5)
    alone, _ = run_sync(backend, [target], max_tokens=16, temperature=0.9, seed=42)
    crowd, _ = run_sync(
        backend,
        [prompt(12, 1), target, prompt(3, 2)],
        max_tokens=16,
        temperature=0.9,
        seed=42,
        max_num_batched_tokens=8,
    )
    assert alone[0].output_ids == crowd[1].output_ids


def test_hf_backend_matches_reference():
    from miniserve.backends.hf import HFBackend
    from miniserve.backends.toy import ByteTokenizer

    _, hf = build_tiny_random(arch="llama", block_size=4)
    backend = HFBackend(hf, ByteTokenizer(), "tiny-hf", kv_cache_mb=1, block_size=4)
    prompts = [prompt(6, 1), prompt(13, 2)]
    seqs, _ = run_sync(backend, prompts, max_tokens=8, max_num_batched_tokens=8)
    for s, p in zip(seqs, prompts, strict=True):
        assert s.output_ids == hf_greedy(hf, p, 8)


def test_unsupported_architecture_is_rejected():
    from transformers import GPT2Config, GPT2LMHeadModel

    from miniserve.backends.native import NativeLlama, UnsupportedModel

    m = GPT2LMHeadModel(GPT2Config(n_layer=1, n_embd=32, n_head=2, vocab_size=50))
    with pytest.raises((UnsupportedModel, AttributeError)):
        NativeLlama(m)
