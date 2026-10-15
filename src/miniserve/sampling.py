"""Token sampling over final-position logits (greedy / temperature / top-p, per-request seeds).

Runs on CPU in float32. Each request gets its own ``torch.Generator`` (seeded if the request
asked for a seed), created on first use and dropped when the request ends, so a seeded request
is reproducible no matter which other requests share its batches — up to floating-point
differences in the logits themselves (batched kernels are not guaranteed bit-identical across
batch compositions; see ARCHITECTURE.md "batch invariance").
"""

from __future__ import annotations

import torch

from miniserve.backends.base import SamplingParams
from miniserve.sequence import Sequence


def sample_next(logits: torch.Tensor, params: SamplingParams, generator: torch.Generator) -> int:
    """Pick the next token from 1-D float32 CPU logits."""
    if params.temperature == 0:
        return int(torch.argmax(logits))
    probs = torch.softmax(logits / params.temperature, dim=-1)
    if params.top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        mass_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
        sorted_probs[mass_before > params.top_p] = 0.0  # always keeps the top token
        choice = torch.multinomial(sorted_probs, 1, generator=generator)
        return int(sorted_idx[choice])
    return int(torch.multinomial(probs, 1, generator=generator))


class Sampler:
    def __init__(self) -> None:
        self._generators: dict[str, torch.Generator] = {}

    def _generator(self, seq: Sequence) -> torch.Generator:
        gen = self._generators.get(seq.request_id)
        if gen is None:
            gen = torch.Generator(device="cpu")
            if seq.params.seed is not None:
                gen.manual_seed(seq.params.seed)
            else:
                gen.seed()
            self._generators[seq.request_id] = gen
        return gen

    def sample(self, logits: torch.Tensor, seqs: list[Sequence]) -> list[int]:
        """``logits``: [len(seqs), vocab] on any device."""
        logits = logits.float().cpu()
        greedy = [i for i, s in enumerate(seqs) if s.params.temperature == 0]
        out = [0] * len(seqs)
        if greedy:  # one vectorised argmax for all greedy rows
            for i, tok in zip(greedy, logits[greedy].argmax(dim=-1).tolist(), strict=True):
                out[i] = tok
        for i, s in enumerate(seqs):
            if s.params.temperature != 0:
                out[i] = sample_next(logits[i], s.params, self._generator(s))
        return out

    def release(self, seq: Sequence, preempted: bool = False) -> None:
        # A preempted request keeps its generator: tokens already sampled are not re-drawn on
        # recompute, so continuing the same random stream keeps seeded output identical.
        if not preempted:
            self._generators.pop(seq.request_id, None)
