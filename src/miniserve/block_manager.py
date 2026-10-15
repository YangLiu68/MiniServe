"""Paged KV-cache block allocator with automatic prefix caching (original implementation).

The KV cache is split into fixed-size blocks of ``block_size`` token slots; each sequence owns a
block table (list of block ids). This removes the need for contiguous per-sequence buffers, so
memory is allocated in small units as sequences grow (the PagedAttention idea, Kwon et al. 2023).

Prefix caching (in the style of vLLM's automatic prefix caching):

* A *full* block is identified by a chained content hash (``Sequence.block_hash``). After its K/V
  are computed the block is registered in ``_cached`` (hash -> block id).
* When a new sequence is admitted, its full prompt blocks are looked up in order; every hit is
  shared (ref count + 1) and its tokens are not recomputed.
* Freed blocks with a registered hash are not wiped: they move to an LRU "evictable" pool and
  can be revived by a later hit. They are only evicted (hash dropped) when a fresh block is
  needed and the plain free list is empty.
* Shared blocks are always full and therefore immutable; sequences only append into their own
  last (partial, unshared) block, so no copy-on-write is needed.

Every block is in exactly one of: ``_free`` (no content), ``_evictable`` (ref 0, cached
content), or in use (ref > 0). ``check_invariants`` asserts this; tests call it after every op.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass

from miniserve.sequence import Sequence


class NoFreeBlocks(RuntimeError):
    pass


@dataclass
class PrefixHit:
    blocks: list[int]
    num_tokens: int


@dataclass
class BlockStats:
    queried_tokens: int = 0
    hit_tokens: int = 0
    evictions: int = 0


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True):
        if num_blocks < 1 or block_size < 1:
            raise ValueError("num_blocks and block_size must be >= 1")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self._ref = [0] * num_blocks
        self._hash: list[bytes | None] = [None] * num_blocks
        self._free: deque[int] = deque(range(num_blocks))
        self._evictable: OrderedDict[int, None] = OrderedDict()  # oldest first
        self._cached: dict[bytes, int] = {}
        self._num_hashed: dict[str, int] = {}  # request_id -> leading blocks already registered
        self.stats = BlockStats()

    # ------------------------------------------------------------------ capacity
    @property
    def num_free_blocks(self) -> int:
        return len(self._free) + len(self._evictable)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - self.num_free_blocks

    @property
    def num_cached_blocks(self) -> int:
        return len(self._cached)

    @property
    def capacity_tokens(self) -> int:
        return self.num_blocks * self.block_size

    def blocks_for(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)

    # ------------------------------------------------------------------ prefix cache
    def match_prefix(self, seq: Sequence) -> PrefixHit:
        """Longest run of cached full blocks at the start of ``seq.token_ids``.

        At least one token is always left uncached so the model has something to run to produce
        the next token's logits.
        """
        if not self.enable_prefix_caching:
            return PrefixHit([], 0)
        max_blocks = (seq.num_tokens - 1) // self.block_size
        blocks: list[int] = []
        for i in range(max_blocks):
            block = self._cached.get(seq.block_hash(i, self.block_size))
            if block is None:
                break
            blocks.append(block)
        return PrefixHit(blocks, len(blocks) * self.block_size)

    def can_admit(self, hit: PrefixHit, num_tokens: int, reserve: int = 0) -> bool:
        """Whether a new sequence can get blocks for ``num_tokens`` given its prefix hit."""
        new_blocks = self.blocks_for(num_tokens) - len(hit.blocks)
        # Reviving an evictable hit takes it out of the free pool too.
        revived = sum(1 for b in hit.blocks if self._ref[b] == 0)
        return new_blocks + revived + reserve <= self.num_free_blocks

    def admit(self, seq: Sequence, hit: PrefixHit, num_tokens: int) -> None:
        """Attach the prefix hit to ``seq`` and allocate blocks up to ``num_tokens``."""
        assert not seq.block_ids, "sequence already holds blocks"
        if not self.can_admit(hit, num_tokens):
            raise NoFreeBlocks(f"cannot admit {seq.request_id}")
        for b in hit.blocks:
            if self._ref[b] == 0:
                del self._evictable[b]
            self._ref[b] += 1
        seq.block_ids = list(hit.blocks)
        self._num_hashed[seq.request_id] = len(hit.blocks)
        if self.enable_prefix_caching:
            self.stats.queried_tokens += seq.num_tokens
            self.stats.hit_tokens += hit.num_tokens
        self.allocate(seq, num_tokens)

    # ------------------------------------------------------------------ allocation
    def can_allocate(self, seq: Sequence, num_tokens: int) -> bool:
        return self.blocks_for(num_tokens) - len(seq.block_ids) <= self.num_free_blocks

    def allocate(self, seq: Sequence, num_tokens: int) -> None:
        """Grow ``seq``'s block table to hold ``num_tokens`` token slots."""
        need = self.blocks_for(num_tokens) - len(seq.block_ids)
        if need > self.num_free_blocks:
            raise NoFreeBlocks(f"{seq.request_id} needs {need}, {self.num_free_blocks} free")
        for _ in range(need):
            b = self._take_fresh_block()
            self._ref[b] = 1
            seq.block_ids.append(b)

    def _take_fresh_block(self) -> int:
        if self._free:
            return self._free.popleft()
        b, _ = self._evictable.popitem(last=False)  # least recently freed
        h = self._hash[b]
        if h is not None and self._cached.get(h) == b:
            del self._cached[h]
        self._hash[b] = None
        self.stats.evictions += 1
        return b

    def commit(self, seq: Sequence) -> None:
        """Register newly *computed* full blocks of ``seq`` in the prefix cache."""
        if not self.enable_prefix_caching:
            return
        done = self._num_hashed.get(seq.request_id, 0)
        full = seq.num_computed // self.block_size
        for i in range(done, full):
            b = seq.block_ids[i]
            h = seq.block_hash(i, self.block_size)
            if h not in self._cached:  # else: identical content computed twice; keep the first
                self._cached[h] = b
                self._hash[b] = h
        self._num_hashed[seq.request_id] = max(done, full)

    def free(self, seq: Sequence) -> None:
        """Release ``seq``'s blocks (tail first, so tails are evicted before shared heads)."""
        for b in reversed(seq.block_ids):
            self._ref[b] -= 1
            assert self._ref[b] >= 0
            if self._ref[b] == 0:
                h = self._hash[b]
                if h is not None and self._cached.get(h) == b:
                    self._evictable[b] = None
                else:
                    self._hash[b] = None
                    self._free.append(b)
        seq.block_ids = []
        self._num_hashed.pop(seq.request_id, None)

    def reset_prefix_cache(self) -> None:
        """Drop all cached-but-unused content (e.g. for cold-cache benchmark runs)."""
        while self._evictable:
            b, _ = self._evictable.popitem(last=False)
            h = self._hash[b]
            if h is not None:
                self._cached.pop(h, None)
            self._hash[b] = None
            self._free.append(b)

    # ------------------------------------------------------------------ debugging
    def check_invariants(self, live: list[Sequence] = ()) -> None:
        free, evict = set(self._free), set(self._evictable)
        assert len(free) == len(self._free), "duplicate in free list"
        assert not free & evict, "block both free and evictable"
        for b in range(self.num_blocks):
            in_use = self._ref[b] > 0
            assert (b in free) + (b in evict) + in_use == 1, f"block {b} in {self._ref[b]=}"
            if b in free:
                assert self._hash[b] is None
            if b in evict:
                assert self._ref[b] == 0 and self._cached.get(self._hash[b]) == b
        for h, b in self._cached.items():
            assert self._hash[b] == h
        if live:
            refs = [0] * self.num_blocks
            for s in live:
                for b in s.block_ids:
                    refs[b] += 1
            assert refs == self._ref, "ref counts disagree with live block tables"
