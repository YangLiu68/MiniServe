### Suite `hol`

M2/M3 - head-of-line blocking. decode-hol: a 'batch' tenant keeps 3 long generations (64->192 tokens) in flight while a 'chat' tenant sends short requests (32->16) as a Poisson process (0.3 req/s); compares one-at-a-time FIFO with continuous batching. prefill-hol: a 'docs' tenant sends 1500-token prompts while a 'chat' tenant streams 96-token answers; compares chunked prefill (256-token step budget) with unchunked prefill (whole prompt in one step).

- Rounds: 2 (config order rotated each round; cells are the median across rounds, <sub>(min–max)</sub> where it varies)
- Host: Apple M4, 16.0 GB, macOS-15.3.2-arm64-arm-64bit; torch 2.14.1, transformers 5.19.0
- Commit: `125472ab17` (dirty); load average at start [10.04, 11.95, 15.7], at end [13.2, 11.74, 13.19]
- Latencies in ms (client-side, streaming). TPOT = (E2E − TTFT)/(tokens − 1).

**decode-hol**

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 | batch | step ms | peak KV | peak RSS MB | preempt | cached |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fifo-sequential | 28/0 | 35.4 <sub>(34.1–36.7)</sub> | 0.44 | 12,142 <sub>(12,019–12,265)</sub> | 15,681 <sub>(15,418–15,943)</sub> | 22.8 | 32.2 | 16,307 | 1.0 | 23.5 | 1% | 552.8 | 0.0 | 0% |
| continuous-chunked | 42/0 | 84.5 <sub>(80.6–88.4)</sub> | 0.73 | 95.8 <sub>(93.3–98.3)</sub> | 217.6 <sub>(163.7–271.6)</sub> | 35.9 | 71.9 | 10,732 | 3.1 | 36.2 | 3% | 502.1 | 0.0 | 0% |
| continuous-unchunked | 42/0 | 84.1 <sub>(76.7–91.5)</sub> | 0.72 | 90.9 <sub>(87.7–94.0)</sub> | 251.5 <sub>(177.7–325.4)</sub> | 36.4 | 60.5 | 10,021 | 3.1 | 36.5 | 4% | 543.5 | 0.0 | 0% |

<sub>tenant `batch`</sub>

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fifo-sequential | 10/0 | 31.9 <sub>(31.5–32.4)</sub> | 0.17 | 11,022 <sub>(10,977–11,067)</sub> | 12,020 <sub>(11,883–12,157)</sub> | 23.2 | 32.3 | 16,290 |
| continuous-chunked | 24/0 | 80.9 <sub>(76.9–84.9)</sub> | 0.42 | 99.6 <sub>(98.5–100.6)</sub> | 170.8 <sub>(157.1–184.6)</sub> | 34.0 | 68.3 | 10,822 |
| continuous-unchunked | 24/0 | 81.2 <sub>(74.8–87.5)</sub> | 0.42 | 125.9 <sub>(106.0–145.9)</sub> | 244.8 <sub>(179.9–309.7)</sub> | 35.8 | 59.5 | 10,067 |

<sub>tenant `chat`</sub>

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fifo-sequential | 18/0 | 5.1 <sub>(5.1–5.2)</sub> | 0.32 | 13,451 <sub>(13,434–13,467)</sub> | 15,827 <sub>(15,608–16,046)</sub> | 22.5 | 30.3 | 16,176 |
| continuous-chunked | 18/0 | 6.5 <sub>(6.5–6.5)</sub> | 0.41 | 86.8 <sub>(84.2–89.3)</sub> | 262.6 <sub>(238.5–286.7)</sub> | 47.1 | 108.7 | 1,604 |
| continuous-unchunked | 18/0 | 6.4 <sub>(6.4–6.5)</sub> | 0.40 | 74.5 <sub>(74.2–74.9)</sub> | 216.2 <sub>(106.0–326.4)</sub> | 43.3 | 63.2 | 1,153 |

**prefill-hol**

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 | batch | step ms | peak KV | peak RSS MB | preempt | cached |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fifo-sequential | 29/0 | 27.9 <sub>(22.7–33.2)</sub> | 0.58 | 4,878 <sub>(4,753–5,003)</sub> | 12,747 <sub>(6,075–19,418)</sub> | 32.0 | 75.4 | 14,242 | 1.0 | 33.4 | 6% | 319.9 | 0.0 | 0% |
| continuous-chunked | 58/0 | 18.3 <sub>(18.1–18.4)</sub> | 1.34 | 1,145 <sub>(1,141–1,150)</sub> | 1,284 <sub>(1,280–1,288)</sub> | 137.7 | 162.1 | 13,009 | 3.3 | 116.7 | 12% | 349.6 | 0.0 | 0% |
| continuous-unchunked | 61/0 | 9.8 <sub>(9.0–10.5)</sub> | 1.39 | 627.0 <sub>(602.2–651.8)</sub> | 1,178 <sub>(1,174–1,182)</sub> | 226.7 | 647.0 | 1,633 | 3.4 | 253.4 | 14% | 401.1 | 0.0 | 0% |

<sub>tenant `chat`</sub>

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fifo-sequential | 14/0 | 27.2 <sub>(22.0–32.3)</sub> | 0.28 | 3,133 <sub>(3,105–3,160)</sub> | 6,809 <sub>(3,598–10,020)</sub> | 24.3 | 72.9 | 13,457 |
| continuous-chunked | 6/0 | 16.6 <sub>(16.5–16.8)</sub> | 0.17 | 554.1 <sub>(548.3–560.0)</sub> | 571.5 <sub>(569.5–573.5)</sub> | 131.2 | 162.7 | 13,484 |
| continuous-unchunked | 2/0 | 8.6 <sub>(7.5–9.8)</sub> | 0.09 | 130.1 <sub>(127.1–133.1)</sub> | 131.0 <sub>(127.6–134.3)</sub> | 236.7 | 662.5 | 22,616 |

<sub>tenant `docs`</sub>

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fifo-sequential | 15/0 | 1.2 <sub>(1.0–1.4)</sub> | 0.30 | 5,318 <sub>(5,293–5,343)</sub> | 13,938 <sub>(6,505–21,371)</sub> | 35.4 | 152.8 | 14,079 |
| continuous-chunked | 52/0 | 5.0 <sub>(5.0–5.1)</sub> | 1.26 | 1,150 <sub>(1,145–1,155)</sub> | 1,287 <sub>(1,285–1,289)</sub> | 137.9 | 158.2 | 1,700 |
| continuous-unchunked | 59/0 | 5.9 <sub>(5.8–6.0)</sub> | 1.47 | 890.6 <sub>(875.6–905.6)</sub> | 1,178 <sub>(1,175–1,182)</sub> | 181.4 | 646.7 | 1,479 |
