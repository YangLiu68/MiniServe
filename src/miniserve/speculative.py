"""Speculative decoding with n-gram ("prompt lookup") drafts — no draft model needed.

A decode step normally produces one token per sequence. At small batch sizes that step is
dominated by fixed costs (kernel launches, reading the weights), so running a few *extra* query
tokens through the same forward pass is almost free. Speculative decoding (Leviathan et al.
2023; Chen et al. 2023) exploits this: guess the next ``k`` tokens cheaply, run
``[last token, guess_1 .. guess_k]`` in one step, and keep the longest prefix of guesses the model
itself would have produced, plus one token from the model's own distribution at the first
mismatch. Every step therefore yields between 1 and ``k + 1`` tokens.

The guesses here come from **n-gram lookup** in the sequence's own context (Saxena 2023, "prompt
lookup decoding"; vLLM's ``ngram`` method): find the most recent earlier occurrence of the last
``n`` tokens and propose what followed it. That is very effective whenever the output copies
from the input — editing, summarising with quotes, RAG answers, code changes — and useless (but
cheap) for free-form text.

Correctness does not depend on the guesses: verification (``sampling.Sampler``) accepts a draft
token only if it is the greedy choice, or — when sampling — with the standard rejection rule
that leaves the output distribution exactly unchanged.

This module is pure Python (no torch): it runs on the engine's worker thread inside the
scheduler, so it is unit-testable without a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from miniserve.sequence import Sequence


@dataclass
class _NgramIndex:
    """For each n: n-gram -> position just after its most recent occurrence.

    Only occurrences that already have a continuation are indexed, so the sequence's current
    suffix never matches itself. Updated incrementally as tokens are appended (O(new tokens)).
    """

    sizes: tuple[int, ...]
    tables: dict[int, dict[tuple[int, ...], int]] = field(default_factory=dict)
    upto: int = 1  # positions e < upto are indexed

    def update(self, tokens: list[int]) -> None:
        for e in range(self.upto, len(tokens)):
            for n in self.sizes:
                if e >= n:
                    self.tables.setdefault(n, {})[tuple(tokens[e - n : e])] = e
        self.upto = max(self.upto, len(tokens))


class NgramProposer:
    """Propose up to ``num_tokens`` draft tokens by matching the last ``ngram_max`` ..
    ``ngram_min`` tokens (longest first) against the sequence's prompt and output."""

    method = "ngram"

    def __init__(self, num_tokens: int = 4, ngram_max: int = 3, ngram_min: int = 2) -> None:
        if num_tokens < 1 or not 1 <= ngram_min <= ngram_max:
            raise ValueError("need num_tokens >= 1 and 1 <= ngram_min <= ngram_max")
        self.num_tokens = num_tokens
        self.sizes = tuple(range(ngram_max, ngram_min - 1, -1))  # longest match first

    def propose(self, seq: Sequence, max_tokens: int | None = None) -> tuple[int, ...]:
        k = self.num_tokens if max_tokens is None else min(self.num_tokens, max_tokens)
        if k <= 0:
            return ()
        index = seq.spec_state
        if index is None:
            index = seq.spec_state = _NgramIndex(self.sizes)
        tokens = seq.token_ids
        index.update(tokens)
        n_tok = len(tokens)
        for n in self.sizes:
            if n_tok < n:
                continue
            end = index.tables.get(n, {}).get(tuple(tokens[n_tok - n :]))
            if end is not None:
                return tuple(tokens[end : end + k])
        return ()
