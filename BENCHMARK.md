# Benchmarks

**Status: no benchmark has been run yet.** Load generation and the benchmark report are
Milestone 6; scheduler comparisons are M2/M3. Nothing below is a performance claim.

## Why there is no M1 baseline

During M1 development the host (Apple M4, 10 cores, 16 GB) was saturated by unrelated processes
(load average 93–121, many busy browser renderer processes). Timings observed during functional
smoke runs are therefore **not representative** and are recorded only so nobody mistakes them for
a baseline later:

| Observation (contaminated, n=1 each) | Value |
|---|---|
| SmolLM2-135M greedy, CPU fp32, warm, 40 tokens | ~5.9 tok/s |
| same, MPS fp16 | ~1.7 tok/s |
| same, MPS fp32 | ~2.3 tok/s |
| Load + warmup, MPS fp16 (server `/readyz` `load_s`) | 61–128 s |
| Tiny random GPT-2 (2 layers, d=64), CPU, steady state | ~2–3 ms/token |

A 135M-parameter model on an M4 should be far faster than this; these numbers reflect CPU and
GPU contention, not MiniServe. What *was* validated is correctness: greedy output is identical
across CPU fp32, MPS fp16 and MPS fp32 for the probe prompt.

## Methodology the real benchmarks will follow (M3/M6)

- Record with every result: git SHA, model id + revision, device, dtype, torch/transformers
  versions, OS, **load average before and after**, and the exact command.
- Refuse (or loudly flag) runs where the 1-minute load average exceeds a threshold relative to
  core count.
- Warmup excluded (server warms up before `/readyz` turns 200); report cold-start separately.
- Client-side metrics from the load generator (TTFT, inter-token latency, end-to-end latency,
  tokens/s, errors) plus server-side `miniserve.timings` for cross-checking.
- Report P50/P95/P99 with sample counts; repeat runs and report spread, not single numbers.
- Comparisons to vLLM/llama.cpp only at matched model, dtype, max tokens, sampling settings and
  hardware class; otherwise label as "different conditions" with the differences listed.
