"""Backend interface shared by the toy backend and the PyTorch/Transformers backend.

A backend owns the tokenizer and the model and exposes generation as a *synchronous* token
iterator. The engine drives that iterator from a dedicated worker thread and checks for
cancellation / deadlines between tokens, so a backend never needs to know about asyncio.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int | None = None


class Tokenizer(Protocol):
    eos_token_ids: frozenset[int]

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str:
        """Decode ids to text, skipping special tokens."""
        ...


class Backend(Protocol):
    name: str
    model_id: str
    device: str
    max_context: int
    tokenizer: Tokenizer

    def generate(self, prompt_ids: list[int], params: SamplingParams) -> Iterator[int]:
        """Yield sampled token ids one at a time (EOS included, if sampled).

        The iterator is lazy: the next forward pass runs only when the caller asks for the next
        token, so a caller that stops iterating stops spending compute. It must not stop on its own
        at ``max_tokens``; the engine enforces limits.
        """
        ...

    def close(self) -> None:
        """Release model memory."""
        ...
