"""Paired A/B on the real model: explicit all-True attention mask vs no mask for a single-sequence
decode step (MPS). Alternates the two variants every iteration and reports the median of the
per-pair ratios, which is robust to slow drift in GPU contention on a shared machine.

    uv run python bench/ab_attention_mask.py results/profile/ab-implicit-mask.json

The committed result also has prefill rows: they were measured with an earlier variant that used
SDPA's is_causal for single-sequence prefill (mixed: -16% at 256 tokens, +6% at 1,024), which
was therefore not kept; with the current code the prefill variants are identical.
"""

import json
import statistics
import sys
import time

import torch

from miniserve.backends.base import SamplingParams
from miniserve.backends.native import NativeBackend, build_metadata
from miniserve.block_manager import BlockManager, PrefixHit
from miniserve.config import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION
from miniserve.sequence import ScheduledSeq, Sequence

be = NativeBackend.from_pretrained(
    DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, "mps", "float16", kv_cache_mb=256
)


def t(meta):
    torch.mps.synchronize()
    a = time.perf_counter()
    be.model.forward(meta, be.kv)
    torch.mps.synchronize()
    return (time.perf_counter() - a) * 1000


out = {}
for name, ctx, n in (
    ("decode ctx=300", 300, 1),
    ("decode ctx=1000", 1000, 1),
    ("prefill 256", 0, 256),
    ("prefill 1024", 0, 1024),
):
    bm = BlockManager(be.num_kv_blocks, be.block_size, False)
    total = ctx + n
    ids = [(j * 31) % 1000 + 100 for j in range(total)]
    s = Sequence(name, ids, SamplingParams(1, temperature=0.0), "t", 0.0, 1e18)
    bm.admit(s, PrefixHit([], 0), total)
    if ctx:
        be.execute([ScheduledSeq(s, 0, ctx)])
    batch = [ScheduledSeq(s, ctx, n)]
    ma = build_metadata(batch, be.block_size, "mps", implicit_mask=False)
    mb = build_metadata(batch, be.block_size, "mps", implicit_mask=True)
    la, lb = be.model.forward(ma, be.kv).float(), be.model.forward(mb, be.kv).float()
    diff = (la - lb).abs().max().item()
    for _ in range(5):
        t(ma)
        t(mb)
    iters = 40 if n == 1 else 12
    ratios, A, B = [], [], []
    for i in range(iters):
        if i % 2:
            b, a = t(mb), t(ma)
        else:
            a, b = t(ma), t(mb)
        A.append(a)
        B.append(b)
        ratios.append(a / b)
    out[name] = dict(
        explicit_ms=round(statistics.median(A), 2),
        implicit_ms=round(statistics.median(B), 2),
        paired_speedup=round(statistics.median(ratios), 3),
        max_logit_diff=diff,
    )
    print(name, out[name], flush=True)
if len(sys.argv) > 1:
    json.dump(out, open(sys.argv[1], "w"), indent=1)
