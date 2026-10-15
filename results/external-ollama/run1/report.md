### Suite `external-ollama`

M5 - same workload against Ollama (llama.cpp + Metal kernels) serving the same model in fp16 (smollm2:135m-instruct-fp16) on the same Mac. Both servers are driven as plain OpenAI clients (greedy, natural EOS, no ignore_eos) because Ollama cannot force output length. Ollama runs with OLLAMA_NUM_PARALLEL=16. See BENCHMARK.md for the (many) caveats.

- Rounds: 2 (config order rotated each round; cells are the median across rounds, <sub>(min–max)</sub> where it varies)
- Host: Apple M4, 16.0 GB, macOS-15.3.2-arm64-arm-64bit; torch 2.14.1, transformers 5.19.0
- Commit: `45b5b4b35b` (dirty); load average at start [11.97, 11.17, 11.07], at end [10.18, 10.71, 10.89]
- Latencies in ms (client-side, streaming). TPOT = (E2E − TTFT)/(tokens − 1).

**external-c1**

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 | batch | step ms | peak KV | peak RSS MB | preempt | cached |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| miniserve-native | 6/0 | 39.4 <sub>(38.8–40.1)</sub> | 0.62 | 47.9 <sub>(47.6–48.2)</sub> | 127.4 <sub>(55.8–198.9)</sub> | 24.5 | 32.8 | 1,763 | 1.0 | 25.1 | 1% | 386.9 | 0.0 | 0% |
| ollama | 6/0 | 112.5 <sub>(108.5–116.6)</sub> | 1.76 | 78.7 <sub>(75.5–81.8)</sub> | 89.6 <sub>(83.7–95.4)</sub> | 8.1 | 9.5 | 610.1 | – | – | – | – | – | 0% |

**external-c4**

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 | batch | step ms | peak KV | peak RSS MB | preempt | cached |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| miniserve-native | 12/0 | 125.1 <sub>(101.7–148.5)</sub> | 2.09 | 143.2 <sub>(103.8–182.5)</sub> | 268.0 <sub>(128.5–407.5)</sub> | 27.5 | 38.9 | 2,267 | 3.7 | 30.2 | 3% | 285.4 | 0.0 | 0% |
| ollama | 12/0 | 163.6 <sub>(144.5–182.8)</sub> | 2.59 | 148.6 <sub>(144.3–153.0)</sub> | 201.8 <sub>(162.8–240.9)</sub> | 21.5 | 31.6 | 1,785 | – | – | – | – | – | 0% |

**external-c16**

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 | batch | step ms | peak KV | peak RSS MB | preempt | cached |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| miniserve-native | 32/0 | 303.6 <sub>(246.8–360.4)</sub> | 4.75 | 238.3 <sub>(205.6–271.0)</sub> | 706.1 <sub>(404.4–1,008)</sub> | 48.2 | 76.9 | 3,775 | 15.3 | 51.6 | 13% | 343.8 | 0.0 | 0% |
| ollama | 32/0 | 242.1 <sub>(228.6–255.6)</sub> | 3.83 | 476.5 <sub>(469.5–483.5)</sub> | 605.7 <sub>(553.9–657.5)</sub> | 59.8 | 95.6 | 5,379 | – | – | – | – | – | 0% |

![external_output_token_throughput](docs/img/bench/external_output_token_throughput.png)

![external_ttft_p95](docs/img/bench/external_ttft_p95.png)

![external_tpot_p50](docs/img/bench/external_tpot_p50.png)
