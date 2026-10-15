"""Speculative decoding on the native paged-KV model (tiny random models, CPU, fp32).

* draft rows: logits for every draft position equal Transformers' logits for that prefix;
* greedy: speculative output is token-identical to plain decoding / the HF reference;
* sampling: the rejection rule leaves the output distribution unchanged (statistical test).
"""

from collections import Counter

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from miniserve.backends.base import SamplingParams  # noqa: E402
from miniserve.backends.native import build_metadata, build_tiny_random  # noqa: E402
from miniserve.block_manager import BlockManager, PrefixHit  # noqa: E402
from miniserve.sampling import sample_next, target_probs, verify_draft  # noqa: E402
from miniserve.sequence import ScheduledSeq, Sequence  # noqa: E402
from miniserve.speculative import NgramProposer  # noqa: E402

from conftest import run_sync  # noqa: E402

pytestmark = pytest.mark.torch
ATOL = 2e-5


@pytest.fixture(scope="module", params=["llama", "qwen2"])
def tiny(request):
    return build_tiny_random(arch=request.param, block_size=4)


def prompt(n, base=7):
    return [(base * 31 + i * 17) % 256 for i in range(n)]


def hf_logits(hf_model, ids):
    with torch.inference_mode():
        return hf_model(torch.tensor([ids])).logits[0, -1].float()


def hf_greedy(hf_model, ids, n):
    ids, out = list(ids), []
    for _ in range(n):
        out.append(int(hf_logits(hf_model, ids).argmax()))
        ids.append(out[-1])
    return out


def forward(backend, batch):
    meta = build_metadata(batch, backend.block_size, backend.device)
    return backend.model.forward(meta, backend.kv).float(), meta


def new_seq(bm, ids, rid, slots):
    s = Sequence(rid, list(ids), SamplingParams(64), "t", 0.0, 1e18)
    bm.admit(s, PrefixHit([], 0), slots)
    return s


def test_draft_rows_match_hf_logits_for_each_prefix(tiny):
    backend, hf = tiny
    bm = BlockManager(backend.num_kv_blocks, backend.block_size, False)
    ids, draft = prompt(11), (5, 77, 200)
    s = new_seq(bm, ids, "spec", len(ids) + len(draft))
    forward(backend, [ScheduledSeq(s, 0, len(ids) - 1)])  # prefill all but the last token
    rows, meta = forward(backend, [ScheduledSeq(s, len(ids) - 1, 1, draft)])
    assert meta.uniform and rows.shape[0] == len(draft) + 1  # one sequence: no padding path
    for j in range(len(draft) + 1):
        want = hf_logits(hf, ids + list(draft[:j]))
        torch.testing.assert_close(rows[j], want, atol=ATOL, rtol=0)


def test_draft_rows_in_a_mixed_ragged_batch(tiny):
    backend, hf = tiny
    bm = BlockManager(backend.num_kv_blocks, backend.block_size, False)
    a_ids, b_ids, c_ids = prompt(9, 1), prompt(14, 2), prompt(6, 3)
    draft = (9, 9)
    a = new_seq(bm, a_ids, "a", len(a_ids) + len(draft))
    b = new_seq(bm, b_ids, "b", len(b_ids))
    c = new_seq(bm, c_ids, "c", len(c_ids))
    forward(backend, [ScheduledSeq(a, 0, 8), ScheduledSeq(b, 0, 5), ScheduledSeq(c, 0, 5)])
    batch = [ScheduledSeq(a, 8, 1, draft), ScheduledSeq(b, 5, 9), ScheduledSeq(c, 5, 1)]
    rows, meta = forward(backend, batch)
    assert not meta.uniform and rows.shape[0] == 3 + 1 + 1
    expected = [hf_logits(hf, a_ids + list(draft[:j])) for j in range(3)]
    expected += [hf_logits(hf, b_ids), hf_logits(hf, c_ids)]
    for row, want in zip(rows, expected, strict=True):
        torch.testing.assert_close(row, want, atol=ATOL, rtol=0)


def test_greedy_speculative_output_is_identical(tiny):
    backend, hf = tiny
    # repetitive prompts so that n-gram drafts are proposed (and sometimes accepted)
    prompts = [prompt(4, 1) * 4, prompt(5, 2) * 3 + prompt(3, 9), prompt(19, 4), [42] * 9]
    base, s0 = run_sync(backend, prompts, max_tokens=24, max_num_batched_tokens=16)
    spec, s1 = run_sync(
        backend,
        prompts,
        max_tokens=24,
        max_num_batched_tokens=16,
        proposer=NgramProposer(4, 3, 1),
    )
    for a, b, p in zip(base, spec, prompts, strict=True):
        assert b.output_ids == a.output_ids == hf_greedy(hf, p, 24)
    assert sum(s.num_draft_tokens for s in spec) > 0


class OracleProposer:
    """Proposes the true continuation, with every ``corrupt_every``-th draft token wrong."""

    method = "oracle"

    def __init__(self, reference: dict[str, list[int]], k: int, corrupt_every: int = 0):
        self.reference, self.num_tokens, self.corrupt_every = reference, k, corrupt_every
        self.calls = 0

    def propose(self, seq, max_tokens=None):
        k = min(self.num_tokens, max_tokens or self.num_tokens)
        g = seq.num_output_tokens
        draft = list(self.reference[seq.request_id][g : g + k])
        self.calls += 1
        if self.corrupt_every and draft and self.calls % self.corrupt_every == 0:
            i = self.calls % len(draft)
            draft[i] = (draft[i] + 1) % 256
        return tuple(draft)


@pytest.mark.parametrize("corrupt_every", [0, 1, 3])
def test_forced_drafts_correct_or_wrong_never_change_greedy_output(tiny, corrupt_every):
    backend, _ = tiny
    prompts = [prompt(7, 1), prompt(12, 2), prompt(3, 3)]
    base, s0 = run_sync(backend, prompts, max_tokens=20)
    ref = {s.request_id: s.output_ids for s in base}
    oracle = OracleProposer(ref, k=4, corrupt_every=corrupt_every)
    spec, s1 = run_sync(backend, prompts, max_tokens=20, proposer=oracle)
    assert [s.output_ids for s in spec] == [s.output_ids for s in base]
    accepted = sum(s.num_accepted_tokens for s in spec)
    if corrupt_every == 0:  # perfect drafts: every draft token accepted, ~5 tokens per step
        assert accepted == sum(s.num_draft_tokens for s in spec) > 0
        assert s1.num_steps <= s0.num_steps / 3
    else:
        assert accepted < sum(s.num_draft_tokens for s in spec)


def test_speculation_with_prefix_cache_and_preemption_is_exact(tiny):
    backend, _ = tiny
    shared = prompt(8, 5)
    prompts = [shared + prompt(4, k) for k in range(4)]
    kw = dict(max_tokens=20, num_blocks=10, max_num_batched_tokens=12)
    base, _ = run_sync(backend, prompts, **kw)
    ref = {s.request_id: s.output_ids for s in base}
    spec, sched = run_sync(backend, prompts, proposer=OracleProposer(ref, 4, 2), **kw)
    assert [s.output_ids for s in spec] == [s.output_ids for s in base]
    assert sched.num_preemptions > 0 and sched.bm.stats.hit_tokens > 0


def test_seeded_speculative_sampling_is_reproducible():
    backend, _ = build_tiny_random(arch="llama", block_size=4, seed=3)
    prompts = [prompt(4, 1) * 3, prompt(6, 2)]
    kw = dict(max_tokens=16, temperature=0.8, seed=7, proposer=NgramProposer(3, 2, 1))
    a, _ = run_sync(backend, prompts, **kw)
    b, _ = run_sync(backend, prompts, **kw)
    assert [s.output_ids for s in a] == [s.output_ids for s in b]


# ---------------------------------------------------------------- the sampling rule


def _tv(counts: Counter, n: int, probs: torch.Tensor) -> float:
    return 0.5 * sum(abs(counts[t] / n - float(probs[t])) for t in range(len(probs)))


@pytest.mark.parametrize("temperature,top_p", [(1.0, 1.0), (0.7, 0.8)])
def test_rejection_sampling_preserves_the_target_distribution(temperature, top_p):
    """Emitted tokens must follow the target distribution whatever the draft: the first token
    ~ p0, and after an accepted first draft token, the second ~ p1. Fixed seeds: deterministic."""
    torch.manual_seed(0)
    V, N = 8, 20_000
    logits = torch.randn(3, V) * 1.5
    params = SamplingParams(10, temperature=temperature, top_p=top_p)
    p = [target_probs(logits[j], params) for j in range(3)]
    draft = (int(p[0].argsort()[-2]), int(p[1].argmax()))  # 2nd most likely, then the top
    gen = torch.Generator().manual_seed(1)
    first, second, n_acc = Counter(), Counter(), 0
    for _ in range(N):
        accepted, tok = verify_draft(logits, draft, params, gen)
        emitted = [*draft[:accepted], tok]
        first[emitted[0]] += 1
        if accepted >= 1:
            second[emitted[1]] += 1
            n_acc += 1
    assert abs(n_acc / N - float(p[0][draft[0]])) < 0.015  # acceptance probability = p0(d)
    assert _tv(first, N, p[0]) < 0.02
    assert _tv(second, n_acc, p[1]) < 0.03


def test_target_probs_is_the_distribution_sample_next_draws_from():
    torch.manual_seed(1)
    logits = torch.randn(10) * 2
    params = SamplingParams(10, temperature=0.9, top_p=0.7)
    gen = torch.Generator().manual_seed(2)
    N = 20_000
    counts = Counter(sample_next(logits, params, gen) for _ in range(N))
    assert _tv(counts, N, target_probs(logits, params)) < 0.02


def test_greedy_verification_accepts_the_argmax_prefix():
    logits = torch.full((4, 5), -1.0)
    for row, best in enumerate([3, 1, 4, 0]):
        logits[row, best] = 5.0
    greedy = SamplingParams(10, temperature=0.0)
    assert verify_draft(logits, (3, 1, 4), greedy, None) == (3, 0)  # all accepted + bonus
    assert verify_draft(logits, (3, 2, 4), greedy, None) == (1, 1)  # stops at the mismatch
    assert verify_draft(logits[:1], (), greedy, None) == (0, 3)
