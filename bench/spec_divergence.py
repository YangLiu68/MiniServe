"""Where greedy speculative and plain decoding diverge on fp16 GPUs, is it a near-tie?

Runs every prompt with speculation off and on (MPS fp16, greedy), finds the first differing
token, then recomputes that position's logits in **fp32 on CPU** with Transformers and reports
the gap between the two candidate tokens. A gap near fp16 rounding (~1e-2 in logit units for
values of this size) means both continuations are valid greedy decodes within the precision of
the run, i.e. a numerical near-tie, not a verification bug (exactness in fp32 is covered by the
test suite).

    uv run python bench/spec_divergence.py --out results/speculative/divergence.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import torch
from spec_sweep import CONFIGS, run_one  # noqa: E402  (bench/ is the script directory)

from miniserve.backends.hf import load_hf_model
from miniserve.backends.native import NativeBackend
from miniserve.config import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="k4")
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--data", default="bench/data/spec_prompts.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    backend = NativeBackend.from_pretrained(
        DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, "mps", "float16", kv_cache_mb=256
    )
    ref, _ = load_hf_model(DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, "cpu", torch.float32)
    tok = backend.tokenizer
    data = json.loads(Path(args.data).read_text())
    report = []
    for set_name in ("copy", "open"):
        for i, text in enumerate(data[set_name]):
            ids = tok.apply_chat_template([{"role": "user", "content": text}])
            a = asyncio.run(run_one(backend, CONFIGS["off"], ids, args.max_tokens))["tokens"]
            b = asyncio.run(run_one(backend, CONFIGS[args.config], ids, args.max_tokens))["tokens"]
            if a == b:
                continue
            k = next(
                (j for j, (x, y) in enumerate(zip(a, b, strict=False)) if x != y),
                min(len(a), len(b)),
            )
            if k >= min(len(a), len(b)):  # one is a prefix of the other: EOS chosen differently
                continue
            with torch.inference_mode():
                logits = ref(torch.tensor([ids + a[:k]])).logits[0, -1].float()
            top2 = torch.topk(logits, 2)
            entry = {
                "set": set_name,
                "prompt": i,
                "position": k,
                "off_token": tok.decode([a[k]]),
                "spec_token": tok.decode([b[k]]),
                "fp32_logit_gap": round(abs(logits[a[k]] - logits[b[k]]).item(), 4),
                "fp32_top2": [tok.decode([t]) for t in top2.indices.tolist()],
                "fp32_top2_gap": round((top2.values[0] - top2.values[1]).item(), 4),
            }
            report.append(entry)
            print(entry, flush=True)
    print(f"{len(report)} divergent outputs")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
