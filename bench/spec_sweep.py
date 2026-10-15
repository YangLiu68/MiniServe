"""Speculative decoding sweep: paired speed comparison, acceptance and exactness per config.

Runs the engine in-process (no HTTP) on the real model, one request at a time (the regime where
speculation pays), greedy, over the prompt sets in ``bench/data/spec_prompts.json``.

**Paired design.** On a shared machine the GPU's speed drifts by 2-4x over minutes (other jobs),
so running each configuration as its own block would mostly measure the drift. Instead every
prompt is run under *all* configurations back to back, in an order rotated per prompt and
round, and the headline is the **per-prompt speedup** time(off) / time(config) — median and
geometric mean over prompts. Prefix caching is off and each request gets a fresh engine, so no
run benefits from another.

Also reported: draft acceptance, tokens produced per decode step, and how many outputs are
identical to the no-speculation baseline (greedy speculative decoding should change speed only;
on fp16 GPUs tiny numeric differences between a 1-token and a k-token verification pass can flip
near-ties, which this measures).

    uv run python bench/spec_sweep.py --device mps --dtype float16 --out results/speculative/sweep
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

from miniserve.backends.base import SamplingParams
from miniserve.backends.native import NativeBackend
from miniserve.config import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION
from miniserve.engine import Engine, EngineConfig, FinishEvent

CONFIGS = {
    "off": dict(speculative="none"),
    "k2": dict(speculative="ngram", num_speculative_tokens=2, ngram_max=3, ngram_min=2),
    "k4": dict(speculative="ngram", num_speculative_tokens=4, ngram_max=3, ngram_min=2),
    "k6": dict(speculative="ngram", num_speculative_tokens=6, ngram_max=3, ngram_min=2),
    "k8": dict(speculative="ngram", num_speculative_tokens=8, ngram_max=3, ngram_min=2),
    "k4-min1": dict(speculative="ngram", num_speculative_tokens=4, ngram_max=3, ngram_min=1),
}


async def run_one(backend, cfg: dict, ids: list[int], max_tokens: int) -> dict:
    engine = Engine(backend, EngineConfig(enable_prefix_caching=False, **cfg))
    engine.start()
    try:
        t0 = time.perf_counter()
        h = engine.submit(ids, SamplingParams(max_tokens=max_tokens, temperature=0.0), 600)
        toks, first = [], None
        async for ev in h.events():
            if isinstance(ev, FinishEvent):
                break
            toks.append(ev.token_id)
            if first is None:
                first = time.perf_counter()
        end = time.perf_counter()
        return {
            "tokens": toks,
            "n": len(toks),
            "e2e_s": end - t0,
            "decode_s": end - first if first else 0.0,
            "steps": engine.num_steps,
            "drafted": engine.spec_draft_tokens,
            "accepted": engine.spec_accepted_tokens,
        }
    finally:
        await engine.stop()


def summarize(runs: list[dict], baseline: list[dict]) -> dict:
    n = sum(r["n"] for r in runs)
    drafted = sum(r["drafted"] for r in runs)
    accepted = sum(r["accepted"] for r in runs)
    decode_steps = sum(r["steps"] - 1 for r in runs)  # the first step is the prefill
    speedups = [b["e2e_s"] / r["e2e_s"] for r, b in zip(runs, baseline, strict=True)]
    return {
        "requests": len(runs),
        "output_tokens": n,
        "tok_per_s": round(n / sum(r["e2e_s"] for r in runs), 2),
        "ms_per_output_token": round(
            1000 * sum(r["decode_s"] for r in runs) / max(1, n - len(runs)), 2
        ),
        "speedup_median": round(statistics.median(speedups), 3),
        "speedup_geomean": round(math.exp(statistics.fmean(math.log(x) for x in speedups)), 3),
        "speedup_min": round(min(speedups), 3),
        "speedup_max": round(max(speedups), 3),
        "tokens_per_decode_step": round((n - len(runs)) / decode_steps, 3)
        if decode_steps
        else None,
        "draft_tokens": drafted,
        "acceptance": round(accepted / drafted, 4) if drafted else None,
        "identical_to_off": sum(
            a["tokens"] == b["tokens"] for a, b in zip(runs, baseline, strict=True)
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--sets", default="copy,open")
    ap.add_argument("--data", default="bench/data/spec_prompts.json")
    ap.add_argument("--out", default="results/speculative/sweep")
    args = ap.parse_args()

    backend = NativeBackend.from_pretrained(
        DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, args.device, args.dtype, kv_cache_mb=256
    )
    tok = backend.tokenizer
    data = json.loads(Path(args.data).read_text())
    sets = {
        name: [tok.apply_chat_template([{"role": "user", "content": p}]) for p in data[name]]
        for name in args.sets.split(",")
    }
    names = args.configs.split(",")
    assert names[0] == "off", "the first config is the baseline"
    for ids in list(sets.values())[0][:2]:  # warm up kernels
        asyncio.run(run_one(backend, CONFIGS["k4"], ids, 16))

    # raw[config][set][round] -> list of per-prompt runs (prompt order)
    raw = {c: {s: [[] for _ in range(args.rounds)] for s in sets} for c in names}
    t_start = time.perf_counter()
    for rnd in range(args.rounds):
        for s, prompts in sets.items():
            for i, ids in enumerate(prompts):
                k = (i + rnd) % len(names)
                order = names[k:] + names[:k]  # rotated per prompt and round
                for c in order:
                    raw[c][s][rnd].append(
                        asyncio.run(run_one(backend, CONFIGS[c], ids, args.max_tokens))
                    )
            print(f"round {rnd} {s:5s} done at {time.perf_counter() - t_start:6.0f}s", flush=True)

    results = {}
    for c in names:
        results[c] = {}
        for s in sets:
            rounds = [summarize(raw[c][s][r], raw["off"][s][r]) for r in range(args.rounds)]
            med = {
                k: statistics.median(x[k] for x in rounds)
                for k in rounds[0]
                if all(isinstance(x[k], int | float) for x in rounds)
            }
            results[c][s] = {"median": med, "rounds": rounds}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = {
        "device": args.device,
        "dtype": args.dtype,
        "max_tokens": args.max_tokens,
        "rounds": args.rounds,
        "configs": {c: CONFIGS[c] for c in names},
        "platform": platform.platform(),
        "loadavg_end": os.getloadavg(),
        "commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
    }
    (out / "results.json").write_text(json.dumps({"env": env, "results": results}, indent=1))
    lines = [
        "| config | prompts | speedup (median, per prompt) | geomean | range | tok/s | "
        "tokens/step | acceptance | identical |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in names:
        for s in sets:
            m = results[c][s]["median"]
            acc = m.get("acceptance")
            lines.append(
                f"| {c} | {s} | {m['speedup_median']:.2f}x | {m['speedup_geomean']:.2f}x | "
                f"{m['speedup_min']:.2f}-{m['speedup_max']:.2f} | {m['tok_per_s']:.1f} | "
                f"{m.get('tokens_per_decode_step') or 0:.2f} | "
                f"{'-' if acc is None else f'{acc:.0%}'} | "
                f"{m['identical_to_off']}/{m['requests']} |"
            )
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
