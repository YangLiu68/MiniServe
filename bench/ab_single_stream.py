"""Paired single-stream decode step: our native forward vs Transformers (DynamicCache).

Both backends share the same weights (fp16 on MPS by default). Each iteration times one decode
step of each, alternating which goes first, at a fixed context length; the headline is the median
of the per-iteration ratios, which is robust to slow drift in GPU contention.

    uv run python bench/ab_single_stream.py --out results/profile/ab-single-stream.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from miniserve.backends.base import SamplingParams
from miniserve.backends.hf import HFBackend, load_hf_model, resolve_dtype
from miniserve.backends.native import NativeBackend
from miniserve.bench.suite import collect_env
from miniserve.block_manager import BlockManager, PrefixHit
from miniserve.config import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION
from miniserve.sequence import ScheduledSeq, Sequence


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--contexts", default="128,512,1024")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dtype = resolve_dtype(args.dtype, args.device)
    hf_model, tok = load_hf_model(DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, args.device, dtype)
    native = NativeBackend.from_hf_model(hf_model, tok, DEFAULT_MODEL_ID, kv_cache_mb=256)
    hf = HFBackend(hf_model, tok, DEFAULT_MODEL_ID, kv_cache_mb=256)

    def sync() -> None:
        if args.device == "mps":
            torch.mps.synchronize()
        elif args.device == "cuda":
            torch.cuda.synchronize()

    def timed(fn) -> float:
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        return (time.perf_counter() - t0) * 1000

    result = {"device": args.device, "dtype": args.dtype, "env": collect_env(), "rows": []}
    for ctx in [int(x) for x in args.contexts.split(",")]:
        ids = [(j * 31) % 1000 + 100 for j in range(ctx + 1)]
        bm = BlockManager(native.num_kv_blocks, native.block_size, False)
        sn = Sequence(f"n{ctx}", ids, SamplingParams(1, temperature=0.0), "t", 0.0, 1e18)
        bm.admit(sn, PrefixHit([], 0), ctx + 1)
        native.execute([ScheduledSeq(sn, 0, ctx)])
        sh = Sequence(f"h{ctx}", ids, SamplingParams(1, temperature=0.0), "t", 0.0, 1e18)
        hf.execute([ScheduledSeq(sh, 0, ctx)])

        def native_step(sn=sn, ctx=ctx):
            native.execute([ScheduledSeq(sn, ctx, 1)])

        def hf_step(sh=sh, ctx=ctx):
            hf._caches[sh.request_id].crop(ctx)  # re-run position `ctx` every time
            hf.execute([ScheduledSeq(sh, ctx, 1)])

        for _ in range(5):
            native_step()
            hf_step()
        n_ms, h_ms, ratios = [], [], []
        for i in range(args.iters):
            if i % 2:
                h, n = timed(hf_step), timed(native_step)
            else:
                n, h = timed(native_step), timed(hf_step)
            n_ms.append(n)
            h_ms.append(h)
            ratios.append(h / n)
        row = {
            "context": ctx,
            "native_ms": round(statistics.median(n_ms), 2),
            "hf_ms": round(statistics.median(h_ms), 2),
            "hf_over_native": round(statistics.median(ratios), 3),
        }
        result["rows"].append(row)
        print(row, flush=True)
        native.release(sn)
        hf.release(sh)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
