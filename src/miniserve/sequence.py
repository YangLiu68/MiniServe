"""Worker-side per-request state shared by the scheduler, block manager and backends.

A ``Sequence`` lives entirely on the engine's worker thread. The asyncio side only ever sees the
``RequestHandle`` (engine.py); the two communicate through a thread-safe inbox and
``call_soon_threadsafe``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from miniserve.backends.base import SamplingParams


class SeqStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass(eq=False)
class Sequence:
    request_id: str
    prompt_ids: list[int]
    params: SamplingParams
    tenant: str
    arrival: float
    deadline: float
    handle: Any = None  # RequestHandle; the worker only reads its cancel flag
    token_ids: list[int] = field(init=False)  # prompt + generated
    num_prompt_tokens: int = field(init=False)
    num_computed: int = 0  # tokens whose K/V are in the cache
    max_computed: int = 0  # high-water mark (recompute after preemption is not re-billed)
    block_ids: list[int] = field(default_factory=list)
    status: SeqStatus = SeqStatus.WAITING
    num_preemptions: int = 0
    num_cached_tokens: int = 0  # prefix-cache hit at first admission
    first_scheduled: float | None = None
    detok: Any = None  # IncrementalDetokenizer, created by the engine worker
    stopper: Any = None  # StopChecker when the request has stop strings
    _block_hashes: list[bytes] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.token_ids = list(self.prompt_ids)
        self.num_prompt_tokens = len(self.prompt_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.token_ids) - self.num_prompt_tokens

    @property
    def output_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens :]

    def block_hash(self, index: int, block_size: int) -> bytes:
        """Chained content hash of full block ``index`` (covers every token up to its end).

        Chaining makes a block's identity depend on its whole prefix, so two prompts that share
        block 3's tokens but differ earlier never alias. blake2b-128 instead of Python's ``hash``:
        a collision here would silently serve another prompt's K/V.
        """
        while len(self._block_hashes) <= index:
            i = len(self._block_hashes)
            parent = self._block_hashes[i - 1] if i else b""
            chunk = self.token_ids[i * block_size : (i + 1) * block_size]
            assert len(chunk) == block_size, "only full blocks are hashed"
            data = b"".join(t.to_bytes(4, "little") for t in chunk)
            self._block_hashes.append(hashlib.blake2b(parent + data, digest_size=16).digest())
        return self._block_hashes[index]


@dataclass(eq=False)
class ScheduledSeq:
    """``num_tokens`` tokens of ``seq`` starting at position ``start`` run in this step."""

    seq: Sequence
    start: int
    num_tokens: int

    @property
    def sample(self) -> bool:
        """True when this chunk reaches the end of the sequence, i.e. a new token is due."""
        return self.start + self.num_tokens == self.seq.num_tokens

    @property
    def token_ids(self) -> list[int]:
        return self.seq.token_ids[self.start : self.start + self.num_tokens]
