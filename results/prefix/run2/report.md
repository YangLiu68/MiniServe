### Suite `prefix`

M4 - automatic prefix caching. Every request shares a 1024-token system prompt and adds a unique ~24-token question (16 output tokens); 4 closed-loop clients. Continuous batching on in both configs.

- Rounds: 2 (config order rotated each round; cells are the median across rounds, <sub>(min–max)</sub> where it varies)
- Host: Apple M4, 16.0 GB, macOS-15.3.2-arm64-arm-64bit; torch 2.14.1, transformers 5.19.0
- Commit: `2411856326`; load average at start [4.35, 4.22, 4.25], at end [5.15, 4.47, 4.33]
- Latencies in ms (client-side, streaming). TPOT = (E2E − TTFT)/(tokens − 1).

**shared-system-prompt**

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 | batch | step ms | peak KV | peak RSS MB | preempt | cached |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prefix-cache-off | 24/0 | 27.4 <sub>(27.3–27.4)</sub> | 1.71 | 818.2 <sub>(808.4–828.1)</sub> | 992.2 <sub>(977.3–1,007)</sub> | 103.4 | 242.9 | 2,643 | 3.6 | 117.4 | 18% | 504.8 | 0.0 | 0% |
| prefix-cache-on | 24/0 | 78.5 <sub>(78.4–78.7)</sub> | 4.91 | 85.7 <sub>(85.1–86.2)</sub> | 218.6 <sub>(218.4–218.7)</sub> | 48.1 | 61.6 | 868.1 | 3.8 | 46.8 | 5% | 467.6 | 0.0 | 94% |
