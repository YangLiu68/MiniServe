### Suite `batching`

M3 - throughput vs latency at matched model/device/dtype (SmolLM2-135M, MPS fp16). Sequential = one request at a time (the M1 design); continuous = iteration-level batching (max 32 sequences, 512-token step budget). Prefix caching off everywhere to isolate batching. Closed-loop clients, prompt 128 tokens, 64 output tokens (ignore_eos).

- Rounds: 2 (config order rotated each round; cells are the median across rounds, <sub>(min–max)</sub> where it varies)
- Host: Apple M4, 16.0 GB, macOS-15.3.2-arm64-arm-64bit; torch 2.14.1, transformers 5.19.0
- Commit: `2411856326`; load average at start [3.99, 4.45, 4.33], at end [4.3, 4.21, 4.24]
- Latencies in ms (client-side, streaming). TPOT = (E2E − TTFT)/(tokens − 1).

**batching-c1**

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 | batch | step ms | peak KV | peak RSS MB | preempt | cached |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hf-sequential | 4/0 | 60.6 <sub>(58.2–63.1)</sub> | 0.95 | 38.6 <sub>(32.2–45.1)</sub> | 49.9 <sub>(47.7–52.0)</sub> | 16.3 | 25.2 | 1,243 | 1.0 | 16.4 | 1% | 440.6 | 0.0 | 0% |
| native-sequential | 4/0 | 72.5 <sub>(70.3–74.7)</sub> | 1.13 | 34.6 <sub>(31.5–37.7)</sub> | 48.1 <sub>(44.9–51.2)</sub> | 12.9 | 20.1 | 1,049 | 1.0 | 13.7 | 1% | 422.4 | 0.0 | 0% |
| native-continuous | 4/0 | 79.3 <sub>(68.8–89.8)</sub> | 1.24 | 26.1 <sub>(23.4–28.9)</sub> | 43.9 <sub>(43.5–44.3)</sub> | 12.1 | 18.9 | 954.6 | 1.0 | 12.7 | 1% | 455.2 | 0.0 | 0% |

**batching-c4**

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 | batch | step ms | peak KV | peak RSS MB | preempt | cached |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hf-sequential | 8/0 | 62.7 <sub>(56.4–69.0)</sub> | 0.98 | 2,967 <sub>(2,548–3,386)</sub> | 3,316 <sub>(2,935–3,697)</sub> | 16.0 | 22.2 | 4,280 | 1.0 | 16.0 | 1% | 325.5 | 0.0 | 0% |
| native-sequential | 8/0 | 67.3 <sub>(65.8–68.8)</sub> | 1.05 | 2,759 <sub>(2,655–2,863)</sub> | 3,036 <sub>(3,034–3,038)</sub> | 14.1 | 20.0 | 3,985 | 1.0 | 14.7 | 1% | 410.1 | 0.0 | 0% |
| native-continuous | 8/0 | 173.2 <sub>(153.6–192.8)</sub> | 2.71 | 95.9 <sub>(84.7–107.1)</sub> | 116.0 <sub>(115.1–117.0)</sub> | 22.3 | 28.8 | 1,640 | 3.9 | 22.8 | 3% | 484.1 | 0.0 | 0% |

**batching-c16**

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 | batch | step ms | peak KV | peak RSS MB | preempt | cached |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hf-sequential | 32/0 | 52.6 <sub>(47.0–58.2)</sub> | 0.82 | 17,977 <sub>(15,569–20,384)</sub> | 19,270 <sub>(16,802–21,739)</sub> | 19.0 | 27.4 | 20,448 | 1.0 | 19.1 | 1% | 219.5 | 0.0 | 0% |
| native-sequential | 32/0 | 70.0 <sub>(69.4–70.6)</sub> | 1.09 | 13,219 <sub>(13,002–13,436)</sub> | 14,286 <sub>(13,912–14,660)</sub> | 13.5 | 20.7 | 15,163 | 1.0 | 14.2 | 1% | 223.4 | 0.0 | 0% |
| native-continuous | 32/0 | 334.7 <sub>(330.5–339.0)</sub> | 5.23 | 235.7 <sub>(233.6–237.9)</sub> | 426.8 <sub>(416.3–437.3)</sub> | 45.2 | 58.1 | 3,333 | 15.3 | 45.1 | 13% | 449.1 | 0.0 | 0% |

![batching_output_token_throughput](docs/img/bench/batching_output_token_throughput.png)

![batching_ttft_p95](docs/img/bench/batching_ttft_p95.png)

![batching_tpot_p50](docs/img/bench/batching_tpot_p50.png)
