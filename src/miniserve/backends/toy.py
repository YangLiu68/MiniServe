"""Deterministic toy backend: no torch, no downloads.

Tokenizer: UTF-8 bytes (ids 0..255) plus EOS = 256.
"Model": echoes the prompt bytes once, then emits EOS. So a request whose ``max_tokens`` is smaller
than the prompt's byte length finishes with ``length``; otherwise with ``stop``. Sampling params
are accepted and ignored.

Speculative decoding is supported: the "model" knows its own next tokens, so draft
verification is exact and cheap, which makes the toy backend a precise test bed for the engine's
multi-token steps.

Step cost is simulated with a sleep, so scheduling experiments get controlled, reproducible
timings (this says nothing about real model performance)::

    step_ms = step_overhead_ms + prefill_ms_per_token * prefill_tokens
              + decode_ms_per_token * decode_tokens

A large ``step_overhead_ms`` with small per-token costs mimics memory-bound GPU decoding, where a
step costs about the same for 1 or 32 sequences — the regime in which batching pays off.
"""

from __future__ import annotations

import time

from miniserve.sequence import ScheduledSeq, Sequence


class ByteTokenizer:
    EOS = 256
    vocab_size = 257

    def __init__(self) -> None:
        self.eos_token_ids = frozenset({self.EOS})

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(i for i in ids if 0 <= i < 256).decode("utf-8", errors="replace")

    def apply_chat_template(self, messages: list[dict[str, str]]) -> list[int]:
        text = "".join(f"{m['role']}: {m['content']}\n" for m in messages) + "assistant: "
        return self.encode(text)


class ToyBackend:
    name = "toy"
    supports_prefix_caching = True
    supports_speculation = True

    def __init__(
        self,
        prefill_ms_per_token: float = 0.0,
        decode_ms_per_token: float = 0.0,
        step_overhead_ms: float = 0.0,
        max_context: int = 1 << 16,
        num_kv_blocks: int = 8192,
        block_size: int = 16,
    ) -> None:
        self.model_id = "toy-echo"
        self.device = "cpu"
        self.dtype = None
        self.max_context = max_context
        self.tokenizer = ByteTokenizer()
        self.block_size = block_size
        self.num_kv_blocks = num_kv_blocks
        self.prefill_s = prefill_ms_per_token / 1000
        self.decode_s = decode_ms_per_token / 1000
        self.overhead_s = step_overhead_ms / 1000

    def execute(self, batch: list[ScheduledSeq]) -> list[int | None]:
        # draft tokens are billed like decode tokens
        decode = sum(
            s.num_query for s in batch if s.num_tokens == 1 and s.start >= s.seq.num_prompt_tokens
        )
        prefill = sum(s.num_query for s in batch) - decode
        cost = self.overhead_s + self.prefill_s * prefill + self.decode_s * decode
        if cost:
            time.sleep(cost)
        return [self._verify(s) if s.sample else None for s in batch]

    @staticmethod
    def _output(seq: Sequence, k: int) -> int:
        """The k-th output token of ``seq``: prompt byte k, then EOS."""
        return seq.token_ids[k] if k < seq.num_prompt_tokens else ByteTokenizer.EOS

    def _verify(self, s: ScheduledSeq) -> int:
        g = s.seq.num_output_tokens
        accepted = 0
        while accepted < len(s.draft) and s.draft[accepted] == self._output(s.seq, g + accepted):
            accepted += 1
        s.accepted = accepted
        return self._output(s.seq, g + accepted)

    def release(self, seq: Sequence, preempted: bool = False) -> None:
        pass

    def memory_stats(self) -> dict[str, float]:
        return {}

    def close(self) -> None:
        pass
