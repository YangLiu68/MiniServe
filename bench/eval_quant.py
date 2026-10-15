"""Quality of the native engine's precision/quantization variants vs an fp32 reference.

Teacher-forced over bench/data/eval_text.txt (original text written for this repo): for every
position, compare next-token distributions of a variant against the Transformers fp32 model.

Reports perplexity, top-1 agreement with the reference and mean KL(ref || variant).

    python -u bench/eval_quant.py --device mps --out results/quant/quality-mps.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from miniserve.backends.base import SamplingParams
from miniserve.backends.hf import load_hf_model, resolve_device
from miniserve.backends.native import NativeBackend, build_metadata
from miniserve.block_manager import BlockManager, PrefixHit
from miniserve.config import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION
from miniserve.sequence import ScheduledSeq, Sequence

TEXT = Path(__file__).parent / "data" / "eval_text.txt"


def windows(ids: list[int], size: int) -> list[list[int]]:
    return [ids[i : i + size] for i in range(0, len(ids) - 1, size) if len(ids[i : i + size]) > 1]


@torch.inference_mode()
def native_all_logits(backend: NativeBackend, ids: list[int]) -> torch.Tensor:
    bm = BlockManager(backend.num_kv_blocks, backend.block_size, False)
    s = Sequence("eval", ids, SamplingParams(1), "t", 0.0, 1e18)
    bm.admit(s, PrefixHit([], 0), len(ids))
    meta = build_metadata([ScheduledSeq(s, 0, len(ids))], backend.block_size, backend.device)
    meta.sample_index = torch.arange(len(ids), device=backend.device)
    return backend.model.forward(meta, backend.kv).float().cpu()


def score(logits: torch.Tensor, ref: torch.Tensor, ids: list[int]) -> dict[str, float]:
    target = torch.tensor(ids[1:])
    lp, ref_lp = F.log_softmax(logits[:-1], -1), F.log_softmax(ref[:-1], -1)
    nll = F.nll_loss(lp, target, reduction="sum").item()
    kl = (ref_lp.exp() * (ref_lp - lp)).sum(-1).sum().item()
    agree = (logits[:-1].argmax(-1) == ref[:-1].argmax(-1)).sum().item()
    return {"nll": nll, "kl": kl, "agree": agree, "n": len(ids) - 1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = resolve_device(args.device)

    ref_model, tok = load_hf_model(DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, "cpu", torch.float32)
    ids_all = tok.encode(TEXT.read_text())
    wins = windows(ids_all, args.window)
    refs = []
    with torch.inference_mode():
        for w in wins:
            refs.append(ref_model(torch.tensor([w])).logits[0].float())

    variants = [("fp32-cpu (native)", "cpu", torch.float32, "none")]
    if device != "cpu":
        variants += [
            (f"fp16-{device}", device, torch.float16, "none"),
            (f"fp16+w8-{device}", device, torch.float16, "int8"),
        ]
    variants += [("fp32+w8-cpu", "cpu", torch.float32, "int8")]
    results = {
        "model": DEFAULT_MODEL_ID,
        "tokens": sum(len(w) - 1 for w in wins),
        "window": args.window,
        "variants": {},
    }
    for name, dev, dtype, quant in variants:
        hf_model, _ = load_hf_model(DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, dev, dtype)
        backend = NativeBackend.from_hf_model(
            hf_model, tok, DEFAULT_MODEL_ID, kv_cache_mb=64, quantization=quant
        )
        del hf_model
        tot = {"nll": 0.0, "kl": 0.0, "agree": 0, "n": 0}
        for w, ref in zip(wins, refs, strict=True):
            for k, v in score(native_all_logits(backend, w), ref, w).items():
                tot[k] += v
        n = tot["n"]
        row = {
            "perplexity": round(float(torch.exp(torch.tensor(tot["nll"] / n))), 3),
            "top1_agreement": round(tot["agree"] / n, 4),
            "mean_kl": round(tot["kl"] / n, 5),
            "weights_mb": round(backend.model.weight_bytes() / 2**20, 1),
        }
        results["variants"][name] = row
        print(
            f"{name:22s} ppl {row['perplexity']:8.3f}  top1 {row['top1_agreement']:.4f}  "
            f"KL {row['mean_kl']:.5f}  weights {row['weights_mb']} MB",
            flush=True,
        )
        backend.close()
    ref_nll = 0.0
    for w, ref in zip(wins, refs, strict=True):
        ref_nll += score(ref, ref, w)["nll"]
    results["reference_fp32_hf_perplexity"] = round(
        float(torch.exp(torch.tensor(ref_nll / results["tokens"]))), 3
    )
    print("reference (Transformers fp32 CPU) ppl", results["reference_fp32_hf_perplexity"])
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
