"""Scheduler behaviour with a fake model (pure Python: no torch, no threads)."""

import random

from miniserve.backends.base import SamplingParams
from miniserve.block_manager import BlockManager
from miniserve.policy import FairPolicy, FCFSPolicy
from miniserve.scheduler import Scheduler, SchedulerConfig
from miniserve.sequence import SeqStatus, Sequence

_ids = iter(range(10**9))


def mkseq(n_prompt, max_tokens=4, tenant="t", arrival=0.0, base=0):
    i = next(_ids)
    toks = [base + (k % 97) for k in range(n_prompt)]
    return Sequence(f"r{i}", toks, SamplingParams(max_tokens), tenant, arrival, 1e9)


def make(num_blocks=64, block_size=4, policy=None, prefix=True, **cfg):
    bm = BlockManager(num_blocks, block_size, enable_prefix_caching=prefix)
    return Scheduler(SchedulerConfig(**cfg), bm, policy if policy is not None else FCFSPolicy())


def step(sched):
    """One engine step with a fake model whose next token is 1000 + position."""
    out = sched.schedule()
    sched.complete_step(out)
    done = []
    for s in out.scheduled:
        if s.sample:
            s.seq.token_ids.append(1000 + s.seq.num_tokens)
            if s.seq.num_output_tokens >= s.seq.params.max_tokens:
                sched.finish(s.seq)
                done.append(s.seq)
    return out, done


def run_to_completion(sched, max_steps=10_000):
    finished, steps = [], 0
    while sched.has_work():
        out, done = step(sched)
        finished += done
        steps += 1
        assert steps < max_steps, "no progress (deadlock?)"
        sched.bm.check_invariants(sched.running)
    return finished, steps


def test_continuous_batching_admits_and_decodes_together():
    sched = make(max_num_batched_tokens=64)
    seqs = [mkseq(5, max_tokens=3) for _ in range(3)]
    for s in seqs:
        sched.add(s)
    out, _ = step(sched)
    assert [s.num_tokens for s in out.scheduled] == [5, 5, 5]  # all prefilled in one step
    out, _ = step(sched)
    assert [s.num_tokens for s in out.scheduled] == [1, 1, 1]  # one batched decode step
    late = mkseq(6, max_tokens=2, base=300)  # distinct tokens: no prefix-cache hit
    sched.add(late)
    out, _ = step(sched)  # newcomer joins the running batch mid-flight
    assert [(s.seq is late, s.num_tokens) for s in out.scheduled] == [
        (False, 1),
        (False, 1),
        (False, 1),
        (True, 6),
    ]
    finished, _ = run_to_completion(sched)
    assert all(s.num_output_tokens == s.params.max_tokens for s in seqs + [late])


def test_chunked_prefill_respects_budget_and_decodes_go_first():
    sched = make(max_num_batched_tokens=8)
    a = mkseq(3, max_tokens=20)
    sched.add(a)
    step(sched)  # a prefilled; now decoding
    long = mkseq(20, max_tokens=2)
    sched.add(long)
    chunks = []
    for _ in range(3):
        out, _ = step(sched)
        assert out.scheduled[0].seq is a and out.scheduled[0].num_tokens == 1  # decode first
        assert out.num_tokens <= 8
        chunks.append(out.scheduled[1].num_tokens)
    assert chunks == [7, 7, 6]  # 20 prompt tokens split under the remaining budget
    assert long.num_output_tokens == 1  # first token sampled only after the last chunk


def test_without_chunked_prefill_long_prompt_waits_for_a_full_budget():
    sched = make(max_num_batched_tokens=20, enable_chunked_prefill=False)
    decoders = [mkseq(2, max_tokens=50) for _ in range(3)]
    for d in decoders:
        sched.add(d)
    step(sched)
    long = mkseq(18, max_tokens=1)
    sched.add(long)
    out, _ = step(sched)  # 3 decodes use 3 tokens, 17 left < 18: long cannot start
    assert all(s.seq is not long for s in out.scheduled)
    for d in decoders[:2]:
        sched.finish(d)
    out, _ = step(sched)  # 1 decode + 18 fits in 20
    assert any(s.seq is long and s.num_tokens == 18 for s in out.scheduled)


def test_max_num_seqs_limits_the_running_batch():
    sched = make(max_num_seqs=2, max_num_batched_tokens=100)
    seqs = [mkseq(3, max_tokens=5) for _ in range(5)]
    for s in seqs:
        sched.add(s)
    out, _ = step(sched)
    assert len(out.scheduled) == 2 and sched.num_waiting == 3
    finished, _ = run_to_completion(sched)
    assert len(finished) == 5


def test_preemption_recomputes_youngest_and_everyone_finishes():
    # 6 blocks x 4 slots = 24 tokens of KV; three sequences that each grow to 4+8=12 tokens
    sched = make(num_blocks=6, block_size=4, max_num_batched_tokens=64, prefix=False)
    seqs = [mkseq(4, max_tokens=8, base=b * 100) for b in range(3)]
    for s in seqs:
        sched.add(s)
    finished, _ = run_to_completion(sched)
    assert sched.num_preemptions > 0
    assert sorted(s.request_id for s in finished) == sorted(s.request_id for s in seqs)
    assert seqs[2].num_preemptions >= 1 and seqs[0].num_preemptions == 0  # youngest goes first
    for s in seqs:  # recompute kept already-generated tokens: outputs follow the fake model
        assert s.output_ids == [1000 + s.num_prompt_tokens + i for i in range(8)]


def test_preempted_sequence_is_readmitted_before_newer_requests():
    sched = make(num_blocks=4, block_size=4, max_num_batched_tokens=64, prefix=False)
    a, b = mkseq(4, max_tokens=6), mkseq(4, max_tokens=6)
    sched.add(a)
    sched.add(b)
    while not sched.num_preemptions:
        step(sched)
    newcomer = mkseq(2, max_tokens=1)
    sched.add(newcomer)
    assert sched.waiting.peek() is b  # preempted b is at the front, ahead of the newcomer


def test_prefix_cache_hit_schedules_only_the_uncached_tail():
    sched = make(max_num_batched_tokens=100)
    system = mkseq(16, max_tokens=1)
    sched.add(system)
    run_to_completion(sched)
    follow = mkseq(16, max_tokens=1)  # identical 16-token prompt (4 full blocks)
    follow.token_ids += [7, 7, 7]
    follow.num_prompt_tokens += 3
    sched.add(follow)
    out, _ = step(sched)
    assert out.scheduled[0].start == 16 and out.scheduled[0].num_tokens == 3
    assert follow.num_cached_tokens == 16


def test_fair_policy_serves_light_tenant_ahead_of_heavy_backlog():
    def order_for(policy):
        sched = make(max_num_seqs=1, max_num_batched_tokens=100, policy=policy)
        for i in range(6):
            sched.add(mkseq(8, max_tokens=4, tenant="heavy", arrival=i))
        sched.add(mkseq(8, max_tokens=4, tenant="light", arrival=10))
        finished, _ = run_to_completion(sched)
        return [s.tenant for s in finished]

    assert order_for(FCFSPolicy()) == ["heavy"] * 6 + ["light"]
    fair = order_for(FairPolicy())
    assert fair.index("light") == 1  # right after the heavy request already in service


def test_fair_policy_does_not_bank_idle_credit():
    p = FairPolicy()
    p.counters = {"busy": 500.0}
    p.push(mkseq(1, tenant="busy"))
    p.push(mkseq(1, tenant="newcomer"))
    assert p.counters["newcomer"] == 500.0  # lifted to the backlogged minimum, not 0


def test_fair_policy_lifts_newcomer_to_the_tenant_being_served():
    """Regression: a tenant arriving into an *empty* queue started at 0 and then won every
    admission against a tenant that had been served alone (banked idle credit)."""
    p = FairPolicy()
    sched = make(policy=p, max_num_seqs=1, max_num_batched_tokens=100)
    for i in range(3):  # A is served alone; nobody waits while its requests run
        sched.add(mkseq(20, max_tokens=4, tenant="A", arrival=i))
        run_to_completion(sched)
    assert p.counters["A"] > 0 and len(p) == 0
    for i in range(6):  # now B shows up, then both keep requests queued
        sched.add(mkseq(20, max_tokens=4, tenant="B", arrival=10 + i))
        sched.add(mkseq(20, max_tokens=4, tenant="A", arrival=10.5 + i))
    assert p.counters["B"] == p.counters["A"]  # lifted, not 0
    finished, _ = run_to_completion(sched)
    first_six = [s.tenant for s in finished[:6]]
    assert first_six.count("A") >= 2, first_six  # interleaved, not BBBBBB


def test_fair_policy_forgets_idle_tenants_and_stays_cheap():
    p = FairPolicy()
    p.max_idle_tenants = 100
    for i in range(5000):  # one-off tenants: each arrives, is admitted, never returns
        s = mkseq(1, tenant=f"t{i}", arrival=i)
        p.push(s)
        assert p.peek() is s
        p.pop(s)
    assert len(p) == 0 and not p._queues
    assert len(p.counters) <= 101


def test_fair_policy_bills_recompute_once():
    p = FairPolicy()
    sched = make(policy=p, max_num_batched_tokens=100)
    s = mkseq(10, max_tokens=2, tenant="x")
    sched.add(s)
    step(sched)
    billed = p.counters["x"]
    assert billed == 10  # 10 input tokens x weight 1
    sched._preempt(s)
    step(sched)  # recompute 11 tokens: only the new output token is billed (weight 2)
    assert p.counters["x"] == billed + 2


def test_randomized_simulation_no_deadlock_no_leaks():
    rng = random.Random(1)
    for policy in (FCFSPolicy(), FairPolicy()):
        sched = make(
            num_blocks=24,
            block_size=4,
            policy=policy,
            max_num_seqs=6,
            max_num_batched_tokens=16,
            watermark_blocks=1,
        )
        live = []
        for i in range(120):
            s = mkseq(
                rng.randint(1, 20),
                max_tokens=rng.randint(1, 30),
                tenant=rng.choice("abc"),
                arrival=i,
                base=rng.choice([0, 0, 500]),
            )
            sched.add(s)
            live.append(s)
            for _ in range(rng.randint(0, 3)):
                step(sched)
            if rng.random() < 0.1:  # random cancellation, queued or running
                victim = rng.choice(live)
                if victim.status != SeqStatus.FINISHED:
                    sched.finish(victim)
            sched.bm.check_invariants(sched.running)
        run_to_completion(sched)
        assert sched.bm.num_used_blocks == 0
        assert all(s.status == SeqStatus.FINISHED for s in live)
