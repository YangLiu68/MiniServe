"""Backend interface: tokenizer + a model that runs one *batched step* at a time.

The engine's scheduler decides which tokens of which sequences run in each step
(``ScheduledSeq``: a contiguous span ``token_ids[start:start+num_tokens]``, which may be a whole
prompt, a prefill chunk, a recompute after preemption, or a single decode token). The backend
runs that span against the sequence's KV cache and, for spans that reach the end of the
sequence, samples the next token. Backends never decide *what* to run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from miniserve.sequence import ScheduledSeq, Sequence


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int | None = None
    ignore_eos: bool = False
    stop: tuple[str, ...] = ()


class Tokenizer(Protocol):
    eos_token_ids: frozenset[int]

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str:
        """Decode ids to text, skipping special tokens."""
        ...

    def apply_chat_template(self, messages: list[dict[str, str]]) -> list[int]: ...


class Backend(Protocol):
    name: str
    model_id: str
    device: str
    dtype: str | None
    max_context: int
    tokenizer: Tokenizer
    block_size: int
    num_kv_blocks: int
    supports_prefix_caching: bool

    def execute(self, batch: list[ScheduledSeq]) -> list[int | None]:
        """Run one step; return the sampled token for each entry whose span reaches the end of
        its sequence (``ScheduledSeq.sample``), else ``None``."""
        ...

    def release(self, seq: Sequence, preempted: bool = False) -> None:
        """Drop per-sequence state (finished, cancelled or preempted)."""
        ...

    def memory_stats(self) -> dict[str, float]: ...

    def close(self) -> None: ...
