### Suite `quant`

M5 - weight-only int8 (W8A16) vs fp16 on the native engine, MPS. Quality is measured separately (bench/eval_quant.py).

- Rounds: 2 (config order rotated each round; cells are the median across rounds, <sub>(min–max)</sub> where it varies)
- Host: Apple M4, 16.0 GB, macOS-15.3.2-arm64-arm-64bit; torch 2.14.1, transformers 5.19.0
- Commit: `9b6a0799fc`; load average at start [10.78, 11.35, 10.61], at end [8.79, 10.59, 10.37]
- Latencies in ms (client-side, streaming). TPOT = (E2E − TTFT)/(tokens − 1).

**quant-c1**

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 | batch | step ms | peak KV | peak RSS MB | preempt | cached |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 4/0 | 47.0 <sub>(44.7–49.3)</sub> | 0.73 | 41.9 <sub>(38.8–45.0)</sub> | 59.4 <sub>(58.9–59.8)</sub> | 20.1 | 27.9 | 1,555 | 1.0 | 21.1 | 1% | 680.3 | 0.0 | 0% |
| fp16+w8 | 4/0 | 55.1 <sub>(50.6–59.6)</sub> | 0.86 | 42.7 <sub>(41.8–43.6)</sub> | 59.4 <sub>(56.3–62.4)</sub> | 17.5 | 23.7 | 1,272 | 1.0 | 18.1 | 1% | 424.8 | 0.0 | 0% |

**quant-c16**

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 | batch | step ms | peak KV | peak RSS MB | preempt | cached |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 32/0 | 368.8 <sub>(354.8–382.8)</sub> | 5.76 | 204.9 <sub>(204.5–205.3)</sub> | 383.6 <sub>(383.1–384.1)</sub> | 41.1 | 52.5 | 3,027 | 15.3 | 41.0 | 13% | 568.5 | 0.0 | 0% |
| fp16+w8 | 32/0 | 382.4 <sub>(372.6–392.3)</sub> | 5.98 | 210.9 <sub>(208.7–213.0)</sub> | 401.8 <sub>(389.1–414.6)</sub> | 38.9 | 49.4 | 2,874 | 15.3 | 39.5 | 13% | 487.4 | 0.0 | 0% |

![quant_output_token_throughput](docs/img/bench/quant_output_token_throughput.png)

![quant_ttft_p95](docs/img/bench/quant_ttft_p95.png)

![quant_tpot_p50](docs/img/bench/quant_tpot_p50.png)
