"""Deterministic toy backend: no torch, no downloads.

Tokenizer: UTF-8 bytes (ids 0..255) plus EOS = 256.
"Model": echoes the prompt bytes once, then emits EOS. So a request whose ``max_tokens`` is smaller
than the prompt's byte length finishes with ``length``; otherwise with ``stop``. Sampling params
are accepted and ignored.

Compute cost is simulated with sleeps (``prefill_ms_per_token`` × prompt length before the first
token, ``decode_ms_per_token`` before each later token). That makes the toy backend useful for
scheduling experiments with controlled, reproducible timings — it says nothing about real model
performance.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

from miniserve.backends.base import SamplingParams


class ByteTokenizer:
    EOS = 256
    vocab_size = 257

    def __init__(self) -> None:
        self.eos_token_ids = frozenset({self.EOS})

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(i for i in ids if 0 <= i < 256).decode("utf-8", errors="replace")


class ToyBackend:
    name = "toy"

    def __init__(
        self,
        prefill_ms_per_token: float = 0.0,
        decode_ms_per_token: float = 0.0,
        max_context: int = 1 << 16,
    ) -> None:
        self.model_id = "toy-echo"
        self.device = "cpu"
        self.max_context = max_context
        self.tokenizer = ByteTokenizer()
        self.prefill_s_per_token = prefill_ms_per_token / 1000
        self.decode_s_per_token = decode_ms_per_token / 1000

    def generate(self, prompt_ids: list[int], params: SamplingParams) -> Iterator[int]:
        if self.prefill_s_per_token:
            time.sleep(self.prefill_s_per_token * len(prompt_ids))
        for i, tok in enumerate([*prompt_ids, ByteTokenizer.EOS]):
            if i > 0 and self.decode_s_per_token:
                time.sleep(self.decode_s_per_token)
            yield tok

    def close(self) -> None:
        pass
