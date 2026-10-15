### Suite `prefix`

M4 - automatic prefix caching. Every request shares a 1024-token system prompt and adds a unique ~24-token question (16 output tokens); 4 closed-loop clients. Continuous batching on in both configs.

- Rounds: 2 (config order rotated each round; cells are the median across rounds, <sub>(min–max)</sub> where it varies)
- Host: Apple M4, 16.0 GB, macOS-15.3.2-arm64-arm-64bit; torch 2.14.1, transformers 5.19.0
- Commit: `45b5b4b35b` (dirty); load average at start [11.31, 10.68, 11.37], at end [12.4, 11.09, 11.47]
- Latencies in ms (client-side, streaming). TPOT = (E2E − TTFT)/(tokens − 1).

**shared-system-prompt**

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 | batch | step ms | peak KV | peak RSS MB | preempt | cached |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prefix-cache-off | 24/0 | 30.4 <sub>(30.1–30.8)</sub> | 1.90 | 671.1 <sub>(658.9–683.3)</sub> | 946.7 <sub>(866.9–1,026)</sub> | 92.7 | 195.3 | 2,446 | 3.6 | 105.5 | 18% | 731.9 | 0.0 | 0% |
| prefix-cache-on | 24/0 | 90.1 <sub>(88.5–91.7)</sub> | 5.63 | 74.9 <sub>(69.5–80.3)</sub> | 220.3 <sub>(219.2–221.3)</sub> | 40.9 | 51.7 | 805.2 | 3.8 | 40.9 | 5% | 708.2 | 0.0 | 94% |
