"""Speculative decoding without torch: n-gram proposer, scheduler, engine and API (toy backend).

The toy backend "knows" its next tokens, so draft verification is exact; the property under test
is that speculation changes *how many steps* a request takes, never *what* it outputs.
"""

import asyncio
import random

import pytest

from miniserve.backends.base import SamplingParams
from miniserve.backends.toy import ByteTokenizer, ToyBackend
from miniserve.block_manager import BlockManager
from miniserve.engine import Engine, EngineConfig, FinishEvent, FinishReason
from miniserve.policy import FairPolicy, FCFSPolicy
from miniserve.scheduler import Scheduler, SchedulerConfig
from miniserve.sequence import Sequence
from miniserve.speculative import NgramProposer

from conftest import RecordingBackend, app_client, make_settings, parse_sse, run_sync


def mkseq(tokens, prompt_len=None, max_tokens=100, rid="s"):
    prompt_len = len(tokens) if prompt_len is None else prompt_len
    s = Sequence(rid, list(tokens[:prompt_len]), SamplingParams(max_tokens), "t", 0.0, 1e18)
    s.token_ids += tokens[prompt_len:]
    return s


# ---------------------------------------------------------------- proposer


def test_proposes_what_followed_the_most_recent_earlier_match():
    #          0  1  2  3  4  5  6  7  8  9 10 11 12 13
    tokens = [1, 2, 3, 4, 5, 9, 1, 2, 3, 7, 8, 1, 2, 3]
    s = mkseq(tokens)
    assert NgramProposer(num_tokens=3, ngram_max=3, ngram_min=1).propose(s) == (7, 8, 1)


def test_longest_ngram_wins_over_a_more_recent_shorter_one():
    tokens = [5, 6, 7, 1, 9, 7, 2, 5, 6, 7]  # (5,6,7) -> 1 9 7 ; most recent (7) -> 2 5 6
    assert NgramProposer(3, ngram_max=3, ngram_min=1).propose(mkseq(tokens)) == (1, 9, 7)
    assert NgramProposer(3, ngram_max=1, ngram_min=1).propose(mkseq(tokens)) == (2, 5, 6)


def test_no_match_short_continuations_and_caps():
    assert NgramProposer(4, 3, 1).propose(mkseq([1, 2, 3, 4])) == ()  # suffix never self-matches
    assert NgramProposer(4, 2, 1).propose(mkseq([1, 1])) == (1,)  # continuation runs out
    s = mkseq([1, 2, 3, 4, 1, 2])
    assert NgramProposer(4, 2, 2).propose(s, max_tokens=0) == ()
    assert NgramProposer(4, 2, 2).propose(s, max_tokens=1) == (3,)
    assert NgramProposer(4, 2, 2).propose(s) == (3, 4, 1, 2)
    with pytest.raises(ValueError):
        NgramProposer(4, ngram_max=1, ngram_min=2)


def _brute_force(tokens, k, sizes):
    for n in sizes:
        if len(tokens) < n:
            continue
        suffix = tokens[-n:]
        for end in range(len(tokens) - 1, n - 1, -1):
            if tokens[end - n : end] == suffix:
                return tuple(tokens[end : end + k])
    return ()


def test_incremental_index_matches_brute_force_search():
    rng = random.Random(0)
    for trial in range(200):
        prop = NgramProposer(rng.randint(1, 6), ngram_max=rng.randint(1, 4), ngram_min=1)
        prop.sizes = tuple(n for n in prop.sizes if n >= rng.randint(1, len(prop.sizes)))
        alphabet = rng.randint(2, 6)
        s = mkseq([rng.randrange(alphabet) for _ in range(rng.randint(1, 30))])
        for _ in range(rng.randint(1, 15)):  # grow the sequence; the index must keep up
            k = rng.choice([None, 1, 2, 3])
            cap = prop.num_tokens if k is None else min(k, prop.num_tokens)
            want = _brute_force(s.token_ids, cap, prop.sizes)
            assert prop.propose(s, k) == want, (trial, s.token_ids)
            s.token_ids += [rng.randrange(alphabet) for _ in range(rng.randint(1, 4))]


# ---------------------------------------------------------------- scheduler (toy model)

PROMPTS = [
    list(b"the cat sat on the mat. the cat sat on the mat. the cat sat"),
    list(b"abcabcabcabcabcabcabcabc"),
    list(b"no repeats here: qwertyuiop"),
    list(b"one two three four. one two three four. five"),
]


def test_toy_outputs_identical_with_fewer_steps():
    toy = ToyBackend()
    base, s0 = run_sync(toy, PROMPTS, max_tokens=40)
    spec, s1 = run_sync(toy, PROMPTS, max_tokens=40, proposer=NgramProposer(4, 3, 2))
    assert [s.output_ids for s in spec] == [s.output_ids for s in base]
    assert sum(s.num_accepted_tokens for s in spec) > 40
    assert s1.num_steps < s0.num_steps


@pytest.mark.parametrize("seed", range(6))
def test_randomized_spec_with_preemption_and_tiny_budget_is_exact(seed):
    """Small KV cache (preemption), small token budget (drafts truncated), prefix caching on,
    block invariants checked after every step by run_sync."""
    rng = random.Random(seed)
    prompts = [
        [rng.choice(b"ab c") for _ in range(rng.randint(3, 30))] for _ in range(rng.randint(2, 7))
    ]
    toy = ToyBackend(num_kv_blocks=64, block_size=4)
    kw = dict(max_tokens=rng.randint(1, 35), num_blocks=24, max_num_batched_tokens=12)
    base, _ = run_sync(toy, prompts, **kw)
    spec, sched = run_sync(
        toy, prompts, proposer=NgramProposer(rng.randint(1, 6), 3, 1), spec_max_batch=8, **kw
    )
    assert [s.output_ids for s in spec] == [s.output_ids for s in base]
    assert sched.bm.num_used_blocks == 0  # nothing leaked


def _decode_ready(n_seqs, **cfg):
    bm = BlockManager(256, 4)
    sched = Scheduler(SchedulerConfig(), bm, FCFSPolicy(), proposer=NgramProposer(4, 2, 1))
    seqs = []
    for i in range(n_seqs):
        s = mkseq([1, 2, 3, 1, 2], max_tokens=50, rid=f"r{i}")
        sched.add(s)
        seqs.append(s)
    out = sched.schedule()  # prefill everything
    sched.complete_step(out)
    for s in seqs:
        s.token_ids.append(3)  # first output token; the suffix (2, 3) now matches the prompt
    for k, v in cfg.items():
        setattr(sched.config, k, v)
    return sched, seqs


def test_speculation_only_for_small_decode_batches():
    sched, _ = _decode_ready(4, spec_max_batch=4)
    assert all(len(s.draft) == 3 for s in sched.schedule().scheduled)  # 3 1 2 after (2, 3)
    sched, _ = _decode_ready(5, spec_max_batch=4)
    assert all(not s.draft for s in sched.schedule().scheduled)


def test_drafts_respect_budget_max_tokens_and_kv_without_preempting():
    # budget 6 for 3 decodes: each keeps 1 token for the others -> drafts shrink
    sched, _ = _decode_ready(3, max_num_batched_tokens=6)
    out = sched.schedule()
    assert out.num_tokens <= 6 and [s.num_tokens for s in out.scheduled] == [1, 1, 1]
    assert out.num_draft_tokens == 3
    # max_tokens: 2 more tokens allowed -> at most 1 draft (the step samples one itself)
    sched, seqs = _decode_ready(1)
    seqs[0].params = SamplingParams(max_tokens=3)
    assert len(sched.schedule().scheduled[0].draft) == 1
    # KV: no free block for draft slots -> the draft is dropped, nobody is preempted
    sched, seqs = _decode_ready(2)
    sched.bm._free.clear()
    sched.bm._evictable.clear()
    out = sched.schedule()
    # 6 tokens in 2 blocks of 4: positions 5 (decode), 6 and 7 (drafts) fit, a 3rd draft not
    assert not out.preempted and [len(s.draft) for s in out.scheduled] == [2, 2]
    assert all(len(s.block_ids) == 2 for s in seqs)


def test_fair_policy_bills_draft_tokens():
    bm = BlockManager(64, 4)
    pol = FairPolicy()
    sched = Scheduler(SchedulerConfig(), bm, pol, proposer=NgramProposer(4, 2, 1))
    s = mkseq([1, 2, 3, 1, 2], max_tokens=50)
    s.tenant = "a"
    sched.add(s)
    sched.complete_step(sched.schedule())
    s.token_ids.append(3)
    before = pol.counters["a"]
    out = sched.schedule()
    assert out.num_draft_tokens == 3
    assert pol.counters["a"] - before == 2.0 * 4  # 1 decode + 3 drafts, output weight 2


# ---------------------------------------------------------------- engine (toy model)


async def _generate(engine, prompt, **params):
    handle = engine.submit(list(prompt.encode()), SamplingParams(**params), 10)
    text, ids, fin = [], [], None
    async for ev in handle.events():
        text.append(ev.text)
        if isinstance(ev, FinishEvent):
            fin = ev
        else:
            ids.append(ev.token_id)
    return "".join(text), ids, fin


CASES = [
    ("the cat sat on the mat. the cat sat on the mat. the cat sat", dict(max_tokens=200)),
    ("the cat sat on the mat. the cat sat on the mat. the cat sat", dict(max_tokens=33)),
    ("abc abc abc abc STOP abc abc abc", dict(max_tokens=200, stop=("STOP",))),
    ("abc abc abc abc abc abc ab", dict(max_tokens=200, stop=("c ab",))),
    ("héllo héllo héllo héllo wörld", dict(max_tokens=200)),
    ("repeat repeat repeat", dict(max_tokens=60, ignore_eos=True)),
]


def test_engine_outputs_identical_and_stop_conditions_hold_inside_drafts():
    async def run(cfg):
        rec = RecordingBackend(ToyBackend())
        engine = Engine(rec, cfg)
        engine.start()
        try:
            results = await asyncio.gather(*(_generate(engine, p, **kw) for p, kw in CASES))
        finally:
            await engine.stop()
        return results, rec.steps, engine

    base, base_steps, _ = asyncio.run(run(EngineConfig(max_num_seqs=1)))
    spec, spec_steps, engine = asyncio.run(
        run(EngineConfig(max_num_seqs=1, speculative="ngram", num_speculative_tokens=5))
    )
    for (bt, bi, bf), (st, si, sf) in zip(base, spec, strict=True):
        assert (st, si, sf.reason, sf.completion_tokens) == (
            bt,
            bi,
            bf.reason,
            bf.completion_tokens,
        )
    assert spec[2][0] == "abc abc abc abc " and spec[2][2].reason == FinishReason.STOP
    assert spec[1][2].completion_tokens == 33 and spec[1][2].reason == FinishReason.LENGTH
    assert sum(f.accepted_tokens for _, _, f in spec) > 50
    assert spec_steps < 0.6 * base_steps
    stats = engine.speculation_stats()
    assert stats["accepted_tokens"] == sum(f.accepted_tokens for _, _, f in spec)
    assert 0 < stats["acceptance_rate"] <= 1


def test_backend_without_verification_disables_speculation():
    class NoSpec(ToyBackend):
        supports_speculation = False

    assert Engine(NoSpec(), EngineConfig(speculative="ngram")).proposer is None
    with pytest.raises(ValueError):
        Engine(ToyBackend(), EngineConfig(speculative="medusa"))


def test_eos_inside_accepted_draft_finishes_with_stop():
    """The prompt contains the EOS id, so an n-gram draft proposes EOS mid-run."""
    eos = ByteTokenizer.EOS

    async def body():
        engine = Engine(ToyBackend(), EngineConfig(speculative="ngram", ngram_min=1))
        engine.start()
        try:
            prompt = list(b"xy") + [eos] + list(b"xy")
            h = engine.submit(prompt, SamplingParams(max_tokens=50), 10)
            toks = [ev.token_id async for ev in h.events() if not isinstance(ev, FinishEvent)]
            return toks, h.finish
        finally:
            await engine.stop()

    toks, fin = asyncio.run(body())
    assert toks == list(b"xy") and fin.reason == FinishReason.STOP


# ---------------------------------------------------------------- HTTP API


def test_api_reports_speculation_and_stream_equals_non_stream():
    body = {"prompt": "to be or not to be, to be or not to be, that is", "max_tokens": 100}
    with app_client(ToyBackend(), speculative="ngram") as c:
        full = c.post("/v1/completions", json=body).json()
        assert full["choices"][0]["text"] == body["prompt"]
        spec = full["miniserve"]["speculative"]
        assert spec["accepted_tokens"] > 10 and spec["draft_tokens"] >= spec["accepted_tokens"]
        chunks, done = parse_sse(c.post("/v1/completions", json={**body, "stream": True}).text)
        assert done and "".join(ch["choices"][0]["text"] for ch in chunks) == body["prompt"]
        state = c.get("/v1/engine").json()["speculative"]
        assert state["method"] == "ngram" and state["accepted_tokens"] > 20
        metrics = c.get("/metrics").text
        assert "miniserve_spec_accepted_tokens_total" in metrics
    with app_client(ToyBackend()) as c:  # off by default
        assert "speculative" not in c.post("/v1/completions", json=body).json()["miniserve"]
        assert c.get("/v1/engine").json()["speculative"] is None


def test_settings_validate_ngram_range():
    with pytest.raises(ValueError):
        make_settings(spec_ngram_min=4, spec_ngram_max=2)
