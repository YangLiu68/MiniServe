"""Token sampling over final-position logits (greedy / temperature / top-p, per-request seeds).

Runs on CPU in float32. Each request gets its own ``torch.Generator`` (seeded if the request
asked for a seed), created on first use and dropped when the request ends, so a seeded request
is reproducible no matter which other requests share its batches — up to floating-point
differences in the logits themselves (batched kernels are not guaranteed bit-identical across
batch compositions; see ARCHITECTURE.md "batch invariance").

Speculative decoding (``verify_draft``) checks draft tokens against these same distributions.
Greedy: accept while the draft equals the argmax. Sampling: the n-gram draft is deterministic
(a point mass q at token d), so the standard rule — accept with probability min(1, p(d)/q(d)) =
p(d), else sample from the residual max(0, p - q) renormalised, i.e. p with d removed — gives
outputs distributed exactly as without speculation (Leviathan et al. 2023, Theorem 1). It uses
the request's random stream differently, so a seeded request is reproducible only for the same
speculation decisions (which depend on load when ``spec_max_batch`` is set).
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


def target_probs(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """The distribution ``sample_next`` draws from (temperature + top-p), normalised."""
    probs = torch.softmax(logits / params.temperature, dim=-1)
    if params.top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        mass_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
        sorted_probs[mass_before > params.top_p] = 0.0
        probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)
    return probs / probs.sum()


def verify_draft(
    logits: torch.Tensor,
    draft: tuple[int, ...],
    params: SamplingParams,
    generator: torch.Generator,
    greedy_ids: list[int] | None = None,
) -> tuple[int, int]:
    """Verify ``draft`` against ``logits`` [len(draft) + 1, vocab] (row j predicts the token
    after draft[:j]). Returns (number of accepted draft tokens, the token that follows them)."""
    m = len(draft)
    if params.temperature == 0:
        best = greedy_ids if greedy_ids is not None else logits.argmax(dim=-1).tolist()
        accepted = 0
        while accepted < m and best[accepted] == draft[accepted]:
            accepted += 1
        return accepted, best[accepted]
    if m == 0:  # identical to the non-speculative path
        return 0, sample_next(logits[0], params, generator)
    for j, d in enumerate(draft):
        p = target_probs(logits[j], params)
        if torch.rand((), generator=generator).item() < p[d].item():
            continue  # accepted with probability p(d)
        p[d] = 0.0  # residual distribution: p without the rejected token
        return j, int(torch.multinomial(p, 1, generator=generator))
    return m, int(torch.multinomial(target_probs(logits[m], params), 1, generator=generator))


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

    def sample_with_drafts(
        self, logits: torch.Tensor, entries: list[tuple[Sequence, tuple[int, ...]]]
    ) -> list[tuple[int, int]]:
        """``logits`` has ``len(draft) + 1`` consecutive rows per entry. Returns
        (accepted draft tokens, next token) per entry."""
        logits = logits.float().cpu()
        greedy_ids = logits.argmax(dim=-1).tolist()  # one vectorised argmax for all rows
        out, row = [], 0
        for seq, draft in entries:
            n = len(draft) + 1
            gen = None if seq.params.temperature == 0 else self._generator(seq)
            out.append(
                verify_draft(
                    logits[row : row + n], draft, seq.params, gen, greedy_ids[row : row + n]
                )
            )
            row += n
        return out

    def release(self, seq: Sequence, preempted: bool = False) -> None:
        # A preempted request keeps its generator: tokens already sampled are not re-drawn on
        # recompute, so continuing the same random stream keeps seeded output identical.
        if not preempted:
            self._generators.pop(seq.request_id, None)
