### Suite `quant`

M5 - weight-only int8 (W8A16) vs fp16 on the native engine, MPS. Quality is measured separately (bench/eval_quant.py).

- Rounds: 2 (config order rotated each round; cells are the median across rounds, <sub>(min–max)</sub> where it varies)
- Host: Apple M4, 16.0 GB, macOS-15.3.2-arm64-arm-64bit; torch 2.14.1, transformers 5.19.0
- Commit: `45b5b4b35b` (dirty); load average at start [11.0, 11.09, 10.52], at end [10.03, 10.73, 10.44]
- Latencies in ms (client-side, streaming). TPOT = (E2E − TTFT)/(tokens − 1).

**quant-c1**

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 | batch | step ms | peak KV | peak RSS MB | preempt | cached |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 4/0 | 34.5 <sub>(32.6–36.4)</sub> | 0.54 | 47.9 <sub>(44.5–51.3)</sub> | 64.9 <sub>(56.8–73.0)</sub> | 26.8 | 42.1 | 2,222 | 1.0 | 28.7 | 1% | 609.7 | 0.0 | 0% |
| fp16+w8 | 4/0 | 24.9 <sub>(23.6–26.3)</sub> | 0.39 | – | – | – | – | 2,826 | 1.0 | 39.8 | 1% | 442.9 | 0.0 | 0% |

**quant-c16**

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 | batch | step ms | peak KV | peak RSS MB | preempt | cached |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 32/0 | 286.8 <sub>(272.4–301.3)</sub> | 4.48 | 245.1 <sub>(231.1–259.0)</sub> | 446.2 <sub>(412.4–480.0)</sub> | 51.8 | 74.9 | 3,839 | 15.3 | 52.7 | 13% | 565.8 | 0.0 | 0% |
| fp16+w8 | 32/0 | 248.4 <sub>(240.7–256.1)</sub> | 3.88 | – | – | – | – | 4,301 | 15.3 | 60.7 | 13% | 293.2 | 0.0 | 0% |

![quant_output_token_throughput](docs/img/bench/quant_output_token_throughput.png)

![quant_ttft_p95](docs/img/bench/quant_ttft_p95.png)

![quant_tpot_p50](docs/img/bench/quant_tpot_p50.png)
