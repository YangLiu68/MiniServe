"""Paged block allocator + prefix cache: accounting invariants, sharing, LRU eviction."""

import random

import pytest

from miniserve.backends.base import SamplingParams
from miniserve.block_manager import BlockManager, NoFreeBlocks
from miniserve.sequence import Sequence

BS = 4


def seq(tokens, rid=None):
    seq.n = getattr(seq, "n", 0) + 1
    return Sequence(rid or f"r{seq.n}", list(tokens), SamplingParams(8), "t", 0.0, 1e9)


def run_prefill(bm, s):
    """Admit with prefix matching and 'compute' the whole prompt."""
    hit = bm.match_prefix(s)
    bm.admit(s, hit, s.num_tokens)
    s.num_computed = s.num_tokens
    bm.commit(s)
    return hit


def test_allocation_grows_by_blocks_and_frees_everything():
    bm = BlockManager(8, BS, enable_prefix_caching=False)
    s = seq(range(10))
    run_prefill(bm, s)
    assert len(s.block_ids) == 3 and bm.num_free_blocks == 5
    for t in range(2):  # decode two tokens: 12 slots still fit in 3 blocks
        s.token_ids.append(100 + t)
        bm.allocate(s, s.num_tokens)
    assert len(s.block_ids) == 3
    s.token_ids.append(7)
    bm.allocate(s, s.num_tokens)  # 13th token needs a 4th block
    assert len(s.block_ids) == 4
    bm.check_invariants([s])
    bm.free(s)
    assert bm.num_free_blocks == 8 and s.block_ids == []
    bm.check_invariants([])


def test_out_of_blocks_raises():
    bm = BlockManager(2, BS)
    s = seq(range(8))
    run_prefill(bm, s)
    s.token_ids.append(1)
    with pytest.raises(NoFreeBlocks):
        bm.allocate(s, s.num_tokens)


def test_shared_prefix_is_reused_and_refcounted():
    bm = BlockManager(16, BS)
    system = list(range(1000, 1012))  # 3 full blocks
    a = seq(system + [1, 2, 3])
    b = seq(system + [7, 8])
    run_prefill(bm, a)
    hit = run_prefill(bm, b)
    assert hit.num_tokens == 12 and hit.blocks == a.block_ids[:3]
    assert b.block_ids[:3] == a.block_ids[:3]  # physically shared
    assert bm.stats.hit_tokens == 12
    bm.check_invariants([a, b])
    bm.free(a)
    bm.check_invariants([b])  # shared blocks still referenced by b
    bm.free(b)
    assert bm.num_cached_blocks == 3  # content survives in the evictable pool
    c = seq(system + [9])
    assert run_prefill(bm, c).num_tokens == 12  # revived from the evictable pool
    bm.check_invariants([c])


def test_readmission_after_preemption_does_not_inflate_hit_stats():
    """Regression: recompute after preemption was counted as a prefix-cache query + hit."""
    bm = BlockManager(16, BS)
    s = seq(range(13))
    run_prefill(bm, s)
    assert (bm.stats.queried_tokens, bm.stats.hit_tokens) == (13, 0)
    bm.free(s)  # preempted: its full blocks stay cached
    s.num_preemptions = 1
    assert run_prefill(bm, s).num_tokens == 12  # recompute reuses its own blocks...
    assert (bm.stats.queried_tokens, bm.stats.hit_tokens) == (13, 0)  # ...but is not a query


def test_cache_salt_isolates_identical_prompts():
    bm = BlockManager(16, BS)
    a = seq(range(9))
    a.cache_salt = b"alice"
    b = seq(range(9))
    b.cache_salt = b"bob"
    c = seq(range(9))
    c.cache_salt = b"alice"
    run_prefill(bm, a)
    assert run_prefill(bm, b).num_tokens == 0  # other tenant: no sharing
    assert run_prefill(bm, c).num_tokens == 8  # same tenant: shared
    bm.check_invariants([a, b, c])


def test_chained_hash_does_not_alias_same_block_at_different_prefix():
    bm = BlockManager(16, BS)
    a = seq([1, 2, 3, 4, 5, 6, 7, 8, 0])
    b = seq([9, 9, 9, 9, 5, 6, 7, 8, 0])  # block 1 has identical tokens, different prefix
    run_prefill(bm, a)
    assert run_prefill(bm, b).num_tokens == 0


def test_full_hit_still_leaves_one_token_to_compute():
    bm = BlockManager(16, BS)
    a = seq(range(8))  # exactly 2 blocks
    run_prefill(bm, a)
    b = seq(range(8))
    hit = bm.match_prefix(b)
    assert hit.num_tokens == 4  # only the first block; last token must be recomputed


def test_lru_evicts_oldest_freed_and_tails_first():
    bm = BlockManager(4, BS)
    a = seq(range(9))  # 3 blocks (2 full + partial)
    run_prefill(bm, a)
    bm.free(a)
    assert bm.num_cached_blocks == 2 and bm.num_free_blocks == 4
    b = seq([50, 51])
    run_prefill(bm, b)  # takes the plain free block first, no eviction
    assert bm.stats.evictions == 0
    c = seq(range(100, 109))  # needs 3 blocks: 1 plain free + both cached ones evicted
    run_prefill(bm, c)
    assert bm.stats.evictions == 2 and bm.num_cached_blocks == 2  # c's own 2 full blocks
    bm.check_invariants([b, c])


def test_eviction_order_keeps_shared_head_longest():
    bm = BlockManager(6, BS)
    a = seq(list(range(12)) + [99])  # 3 full blocks + 1 token
    run_prefill(bm, a)
    bm.free(a)  # tail-first: full blocks enter the LRU as block2, block1, block0
    filler = seq(range(500, 513))  # 4 blocks: 3 plain free (2 + a's partial), then 1 eviction
    run_prefill(bm, filler)
    assert bm.stats.evictions == 1
    probe = seq(list(range(12)) + [1])
    assert bm.match_prefix(probe).num_tokens == 8  # head blocks 0,1 survived; tail evicted


def test_can_admit_counts_revived_blocks():
    bm = BlockManager(3, BS)
    a = seq(range(8))
    a.token_ids.append(0)
    run_prefill(bm, a)  # 3 blocks
    bm.free(a)  # 2 cached (evictable) + 1 free
    b = seq(list(range(8)) + [5, 6, 7, 8, 9])  # hit 2 blocks, needs 2 more (13 tokens -> 4)
    hit = bm.match_prefix(b)
    assert hit.num_tokens == 8
    assert not bm.can_admit(hit, b.num_tokens)  # 2 new + 2 revived > 3 free
    assert bm.can_admit(hit, 12)  # 1 new + 2 revived == 3


def test_randomized_workload_keeps_invariants():
    rng = random.Random(0)
    bm = BlockManager(40, BS)
    prefixes = [list(range(p * 100, p * 100 + rng.randint(4, 16))) for p in range(4)]
    live = []
    for _ in range(600):
        op = rng.random()
        if op < 0.45 or not live:
            toks = rng.choice(prefixes) + [rng.randint(0, 50) for _ in range(rng.randint(1, 9))]
            s = seq(toks)
            hit = bm.match_prefix(s)
            if bm.can_admit(hit, s.num_tokens):
                bm.admit(s, hit, s.num_tokens)
                s.num_computed = s.num_tokens
                bm.commit(s)
                live.append(s)
        elif op < 0.8:
            s = rng.choice(live)
            s.token_ids.append(rng.randint(0, 50))
            if bm.can_allocate(s, s.num_tokens):
                bm.allocate(s, s.num_tokens)
                s.num_computed = s.num_tokens
                bm.commit(s)
            else:
                s.token_ids.pop()
        else:
            s = live.pop(rng.randrange(len(live)))
            bm.free(s)
        bm.check_invariants(live)
    assert bm.stats.hit_tokens > 0 and bm.stats.evictions > 0
    for s in live:
        bm.free(s)
    bm.check_invariants([])
    assert bm.num_free_blocks == 40


def test_reset_prefix_cache():
    bm = BlockManager(8, BS)
    a = seq(range(9))
    run_prefill(bm, a)
    bm.free(a)
    assert bm.num_cached_blocks == 2
    bm.reset_prefix_cache()
    assert bm.num_cached_blocks == 0 and bm.match_prefix(seq(range(9))).num_tokens == 0
    bm.check_invariants([])
