### Suite `batching`

M3 - throughput vs latency at matched model/device/dtype (SmolLM2-135M, MPS fp16). Sequential = one request at a time (the M1 design); continuous = iteration-level batching (max 32 sequences, 512-token step budget). Prefix caching off everywhere to isolate batching. Closed-loop clients, prompt 128 tokens, 64 output tokens (ignore_eos).

- Rounds: 2 (config order rotated each round; cells are the median across rounds, <sub>(min–max)</sub> where it varies)
- Host: Apple M4, 16.0 GB, macOS-15.3.2-arm64-arm-64bit; torch 2.14.1, transformers 5.19.0
- Commit: `be441d8655` (dirty); load average at start [14.53, 12.8, 13.93], at end [11.21, 11.7, 12.82]
- Latencies in ms (client-side, streaming). TPOT = (E2E − TTFT)/(tokens − 1).

**batching-c1**

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 | batch | step ms | peak KV | peak RSS MB | preempt | cached |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hf-sequential | 4/0 | 28.7 <sub>(25.7–31.7)</sub> | 0.45 | 66.8 <sub>(54.5–79.1)</sub> | 86.0 <sub>(76.2–95.9)</sub> | 34.7 | 48.4 | 2,461 | 1.0 | 34.9 | 1% | 423.7 | 0.0 | 0% |
| native-sequential | 4/0 | 25.0 <sub>(23.8–26.2)</sub> | 0.39 | 75.1 <sub>(56.5–93.7)</sub> | 100.1 <sub>(65.1–135.1)</sub> | 37.9 | 55.4 | 2,953 | 1.0 | 39.7 | 1% | 463.9 | 0.0 | 0% |
| native-continuous | 4/0 | 28.0 <sub>(24.4–31.6)</sub> | 0.44 | 70.6 <sub>(45.2–96.1)</sub> | 99.3 <sub>(83.0–115.6)</sub> | 35.9 | 45.9 | 2,480 | 1.0 | 36.0 | 1% | 471.4 | 0.0 | 0% |

**batching-c4**

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 | batch | step ms | peak KV | peak RSS MB | preempt | cached |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hf-sequential | 8/0 | 29.3 <sub>(26.0–32.5)</sub> | 0.46 | 5,942 <sub>(5,812–6,071)</sub> | 7,038 <sub>(6,275–7,800)</sub> | 32.8 | 51.5 | 9,568 | 1.0 | 34.3 | 1% | 267.3 | 0.0 | 0% |
| native-sequential | 8/0 | 25.5 <sub>(24.4–26.7)</sub> | 0.40 | 7,453 <sub>(7,062–7,843)</sub> | 8,021 <sub>(7,892–8,150)</sub> | 38.4 | 53.3 | 10,431 | 1.0 | 39.0 | 1% | 302.4 | 0.0 | 0% |
| native-continuous | 8/0 | 88.1 <sub>(72.8–103.5)</sub> | 1.38 | 264.6 <sub>(104.2–425.0)</sub> | 396.4 <sub>(174.0–618.8)</sub> | 43.4 | 50.5 | 3,338 | 3.9 | 45.6 | 3% | 379.0 | 0.0 | 0% |

**batching-c16**

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 | batch | step ms | peak KV | peak RSS MB | preempt | cached |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hf-sequential | 32/0 | 29.2 <sub>(23.2–35.2)</sub> | 0.46 | 32,409 <sub>(23,897–40,921)</sub> | 38,436 <sub>(32,185–44,686)</sub> | 33.6 | 51.9 | 40,526 | 1.0 | 35.5 | 1% | 224.3 | 0.0 | 0% |
| native-sequential | 32/0 | 23.8 <sub>(23.6–24.1)</sub> | 0.37 | 38,314 <sub>(37,635–38,993)</sub> | 42,218 <sub>(41,449–42,986)</sub> | 39.9 | 59.6 | 44,805 | 1.0 | 41.7 | 1% | 246.3 | 0.0 | 0% |
| native-continuous | 32/0 | 197.3 <sub>(193.1–201.5)</sub> | 3.08 | 463.1 <sub>(403.9–522.4)</sub> | 765.3 <sub>(569.0–961.6)</sub> | 73.2 | 120.3 | 5,986 | 15.3 | 76.6 | 13% | 348.9 | 0.0 | 0% |

![batching_output_token_throughput](docs/img/bench/batching-run1/batching_output_token_throughput.png)

![batching_ttft_p95](docs/img/bench/batching-run1/batching_ttft_p95.png)

![batching_tpot_p50](docs/img/bench/batching-run1/batching_tpot_p50.png)
