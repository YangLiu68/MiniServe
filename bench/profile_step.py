"""Micro-benchmark: model step latency vs batch size (decode) and vs prompt length (prefill).

Explains *why* continuous batching helps (or not) on a given device: if a decode step for 16
sequences costs about the same as for 1, batching multiplies throughput. Measures the native
paged backend and, for reference, the Transformers DynamicCache backend.

    python bench/profile_step.py --device mps --dtype float16 --out results/profile/mps-fp16.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from miniserve.backends.base import SamplingParams
from miniserve.backends.hf import HFBackend, load_hf_model, resolve_device, resolve_dtype
from miniserve.backends.native import NativeBackend
from miniserve.bench.suite import collect_env
from miniserve.block_manager import BlockManager, PrefixHit
from miniserve.config import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION
from miniserve.sequence import ScheduledSeq, Sequence


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def timed(fn, device: str, iters: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    sync(device)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync(device)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1000


def make_seqs(backend, bm: BlockManager, n: int, ctx: int) -> list[Sequence]:
    seqs = []
    for i in range(n):
        ids = [(i * 7919 + j * 31) % 1000 + 100 for j in range(ctx + 1)]
        s = Sequence(f"p{i}", ids, SamplingParams(1, temperature=0.0), "t", 0.0, 1e18)
        bm.admit(s, PrefixHit([], 0), ctx + 1)
        seqs.append(s)
    # fill the cache with a real prefill so attention reads meaningful values
    backend.execute([ScheduledSeq(s, 0, ctx) for s in seqs])
    return seqs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--batches", default="1,2,4,8,16,32,64")
    ap.add_argument("--context", type=int, default=256)
    ap.add_argument("--prefill", default="64,256,1024")
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--hf-batches", default="1,4,16")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    args.device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, args.device)
    args.dtype = str(dtype).removeprefix("torch.")
    hf_model, tok = load_hf_model(DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, args.device, dtype)
    native = NativeBackend.from_hf_model(hf_model, tok, DEFAULT_MODEL_ID, kv_cache_mb=512)
    hf = HFBackend(hf_model, tok, DEFAULT_MODEL_ID, kv_cache_mb=512)
    result = {
        "device": args.device,
        "dtype": args.dtype,
        "context": args.context,
        "env": collect_env(),
        "decode": [],
        "prefill": [],
        "hf_decode": [],
    }

    for b in [int(x) for x in args.batches.split(",")]:
        bm = BlockManager(native.num_kv_blocks, native.block_size, False)
        seqs = make_seqs(native, bm, b, args.context)
        batch = [ScheduledSeq(s, args.context, 1) for s in seqs]
        ms = timed(lambda batch=batch: native.execute(batch), args.device, args.iters)
        result["decode"].append(
            {"batch": b, "step_ms": round(ms, 2), "tokens_per_s": round(b / ms * 1000, 1)}
        )
        print(f"native decode  B={b:3d}: {ms:7.2f} ms/step  {b / ms * 1000:8.1f} tok/s")
        for s in seqs:
            native.release(s)

    for n in [int(x) for x in args.prefill.split(",")]:
        bm = BlockManager(native.num_kv_blocks, native.block_size, False)
        ids = [(j * 31) % 1000 + 100 for j in range(n)]
        s = Sequence("pf", ids, SamplingParams(1, temperature=0.0), "t", 0.0, 1e18)
        bm.admit(s, PrefixHit([], 0), n)
        batch = [ScheduledSeq(s, 0, n)]
        ms = timed(lambda batch=batch: native.execute(batch), args.device, max(3, args.iters // 3))
        result["prefill"].append(
            {"tokens": n, "step_ms": round(ms, 2), "tokens_per_s": round(n / ms * 1000, 1)}
        )
        print(f"native prefill N={n:5d}: {ms:7.2f} ms  {n / ms * 1000:9.1f} tok/s")

    for b in [int(x) for x in args.hf_batches.split(",")]:
        seqs = []
        for i in range(b):
            ids = [(i * 7919 + j * 31) % 1000 + 100 for j in range(args.context + 1)]
            s = Sequence(f"h{i}", ids, SamplingParams(1, temperature=0.0), "t", 0.0, 1e18)
            hf.execute([ScheduledSeq(s, 0, args.context)])
            seqs.append(s)

        def step(seqs=seqs):
            # re-run position `context` against a cache cropped back to `context` each time
            for s in seqs:
                hf._caches[s.request_id].crop(args.context)
            hf.execute([ScheduledSeq(s, args.context, 1) for s in seqs])

        ms = timed(step, args.device, args.iters)
        result["hf_decode"].append(
            {"batch": b, "step_ms": round(ms, 2), "tokens_per_s": round(b / ms * 1000, 1)}
        )
        print(f"hf     decode  B={b:3d}: {ms:7.2f} ms/step  {b / ms * 1000:8.1f} tok/s")
        for s in seqs:
            hf.release(s)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
