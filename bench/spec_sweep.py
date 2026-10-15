"""Speculative decoding sweep: speed, acceptance and exactness per draft configuration.

Runs the engine in-process (no HTTP) on the real model, one request at a time (the regime where
speculation pays), greedy, over the prompt sets in ``bench/data/spec_prompts.json``. For every
configuration it reports output tokens/s, mean time per output token, draft acceptance, tokens
produced per decode step, and how many outputs are identical to the no-speculation baseline
(greedy speculative decoding should change speed only; on fp16 GPUs tiny numeric differences
between a 1-token and a k-token verification pass can flip near-ties, which this measures).

    uv run python bench/spec_sweep.py --device mps --dtype float16 --out results/speculative/sweep

Configs run in rotated order across rounds so that drift on a shared machine does not favour
any one of them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
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


async def run_set(backend, cfg: dict, prompts: list[list[int]], max_tokens: int) -> list[dict]:
    engine = Engine(backend, EngineConfig(max_num_seqs=4, **cfg))
    engine.start()
    out = []
    try:
        for ids in prompts:
            steps0, drafted0, acc0 = (
                engine.num_steps,
                engine.spec_draft_tokens,
                engine.spec_accepted_tokens,
            )
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
            out.append(
                {
                    "tokens": toks,
                    "n": len(toks),
                    "e2e_s": end - t0,
                    "decode_s": end - first if first else 0.0,
                    "steps": engine.num_steps - steps0,
                    "drafted": engine.spec_draft_tokens - drafted0,
                    "accepted": engine.spec_accepted_tokens - acc0,
                }
            )
    finally:
        await engine.stop()
    return out


def summarize(runs: list[dict], baseline: list[dict] | None) -> dict:
    n = sum(r["n"] for r in runs)
    decode_s = sum(r["decode_s"] for r in runs)
    drafted = sum(r["drafted"] for r in runs)
    accepted = sum(r["accepted"] for r in runs)
    decode_steps = sum(r["steps"] - 1 for r in runs)  # first step is the prefill
    s = {
        "requests": len(runs),
        "output_tokens": n,
        "e2e_s_total": round(sum(r["e2e_s"] for r in runs), 3),
        "tok_per_s": round(n / sum(r["e2e_s"] for r in runs), 2),
        "ms_per_output_token": round(1000 * decode_s / max(1, n - len(runs)), 2),
        "tokens_per_decode_step": round((n - len(runs)) / decode_steps, 3)
        if decode_steps
        else None,
        "draft_tokens": drafted,
        "acceptance": round(accepted / drafted, 4) if drafted else None,
    }
    if baseline is not None:
        s["identical_to_off"] = sum(
            a["tokens"] == b["tokens"] for a, b in zip(runs, baseline, strict=True)
        )
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--configs", default=",".join(CONFIGS))
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
        for name in ("copy", "open")
    }
    names = args.configs.split(",")
    asyncio.run(run_set(backend, CONFIGS["off"], sets["open"][:2], 8))  # warm up kernels

    raw: dict[str, dict[str, list[list[dict]]]] = {c: {s: [] for s in sets} for c in names}
    for rnd in range(args.rounds):
        order = names[rnd % len(names) :] + names[: rnd % len(names)]
        if rnd % 2:
            order.reverse()
        for c in order:
            for s, prompts in sets.items():
                t = time.perf_counter()
                raw[c][s].append(
                    asyncio.run(run_set(backend, CONFIGS[c], prompts, args.max_tokens))
                )
                print(f"round {rnd} {c:8s} {s:5s} {time.perf_counter() - t:6.1f}s", flush=True)

    results = {}
    for c in names:
        results[c] = {}
        for s in sets:
            rounds = [
                summarize(r, raw["off"][s][i] if "off" in raw else None)
                for i, r in enumerate(raw[c][s])
            ]
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
        "platform": platform.platform(),
        "loadavg_end": os.getloadavg(),
        "commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
    }
    (out / "results.json").write_text(json.dumps({"env": env, "results": results}, indent=1))
    lines = [
        "| config | prompts | tok/s | ms/token | tokens/step | acceptance | identical |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for c in names:
        for s in sets:
            m = results[c][s]["median"]
            acc = m.get("acceptance")
            lines.append(
                f"| {c} | {s} | {m['tok_per_s']:.1f} | {m['ms_per_output_token']:.1f} | "
                f"{m.get('tokens_per_decode_step') or 0:.2f} | "
                f"{'–' if acc is None else f'{acc:.0%}'} | "
                f"{m.get('identical_to_off', '–')}/{m['requests']} |"
            )
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
