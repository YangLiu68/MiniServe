### Suite `fairness`

M2 - fairness under contention. Capacity is limited to 4 running sequences; tenant 'heavy' keeps 12 requests in flight, tenant 'light' keeps 1 (64->64 tokens each). FCFS serves in arrival order; fair = Virtual Token Counter (Sheng et al., OSDI'24).

- Rounds: 2 (config order rotated each round; cells are the median across rounds, <sub>(min–max)</sub> where it varies)
- Host: Apple M4, 16.0 GB, macOS-15.3.2-arm64-arm-64bit; torch 2.14.1, transformers 5.19.0
- Commit: `45b5b4b35b` (dirty); load average at start [18.47, 14.55, 12.77], at end [9.15, 11.42, 11.85]
- Latencies in ms (client-side, streaming). TPOT = (E2E − TTFT)/(tokens − 1).

**heavy-vs-light**

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 | batch | step ms | peak KV | peak RSS MB | preempt | cached |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fcfs | 102/0 | 104.8 <sub>(99.6–110.0)</sub> | 1.64 | 4,770 <sub>(4,713–4,826)</sub> | 7,279 <sub>(7,027–7,530)</sub> | 34.0 | 44.2 | 9,536 | 3.9 | 34.0 | 2% | 686.6 | 0.0 | 0% |
| fair-vtc | 108/0 | 110.6 <sub>(110.4–110.7)</sub> | 1.73 | 4,662 <sub>(4,645–4,678)</sub> | 7,666 <sub>(7,572–7,761)</sub> | 32.4 | 46.6 | 10,088 | 3.9 | 31.6 | 2% | 747.7 | 0.0 | 0% |

<sub>tenant `heavy`</sub>

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fcfs | 94/0 | 97.1 <sub>(92.3–101.8)</sub> | 1.52 | 4,809 <sub>(4,713–4,905)</sub> | 7,218 <sub>(7,019–7,417)</sub> | 34.1 | 44.3 | 9,472 |
| fair-vtc | 94/0 | 96.8 <sub>(96.4–97.1)</sub> | 1.51 | 4,961 <sub>(4,760–5,161)</sub> | 7,751 <sub>(7,626–7,876)</sub> | 32.4 | 46.6 | 10,271 |

<sub>tenant `light`</sub>

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fcfs | 8/0 | 9.2 <sub>(8.8–9.5)</sub> | 0.14 | 4,609 <sub>(4,553–4,665)</sub> | 6,677 <sub>(6,513–6,842)</sub> | 33.2 | 42.9 | 8,974 |
| fair-vtc | 14/0 | 15.4 <sub>(15.3–15.5)</sub> | 0.24 | 2,135 <sub>(2,088–2,181)</sub> | 2,723 <sub>(2,650–2,795)</sub> | 31.9 | 46.3 | 5,316 |
