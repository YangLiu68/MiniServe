### Suite `speculative`

n-gram (prompt lookup) speculative decoding vs plain decoding, SmolLM2-135M on MPS fp16, greedy (outputs are the same text in both configs, so per-token latency is directly comparable). 'copy' prompts ask for edits/reformatting of a given text; 'open' prompts are free-form. Speculation is used only while <= 4 sequences decode, so the 16-client scenario checks that it switches off without a regression. Prefix caching off to isolate the effect.

- Rounds: 2 (config order rotated each round; cells are the median across rounds, <sub>(min–max)</sub> where it varies)
- Host: Apple M4, 16.0 GB, macOS-15.3.2-arm64-arm-64bit; torch 2.14.1, transformers 5.19.0
- Commit: `1b25edf8e4` (dirty); load average at start [13.03, 26.87, 36.71], at end [14.67, 18.25, 19.34]
- Latencies in ms (client-side, streaming). TPOT = (E2E − TTFT)/(tokens − 1).

**copy-c1**

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 | batch | step ms | peak KV | peak RSS MB | preempt | cached | draft accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spec-off | 12/0 | 27.0 <sub>(22.4–31.5)</sub> | 0.11 | 90.5 <sub>(85.0–95.9)</sub> | 204.8 <sub>(115.5–294.1)</sub> | 33.7 | 81.6 | 16,561 | 1.0 | 37.8 | 2% | 486.1 | 0.0 | 0% | – |
| spec-ngram | 12/0 | 37.1 <sub>(35.4–38.7)</sub> | 0.14 | 79.0 <sub>(75.2–82.7)</sub> | 95.5 <sub>(93.8–97.1)</sub> | 27.5 | 48.6 | 8,080 | 1.0 | 33.7 | 2% | 457.8 | 0.0 | 0% | 32% |

**open-c1**

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 | batch | step ms | peak KV | peak RSS MB | preempt | cached | draft accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spec-off | 12/0 | 24.2 <sub>(13.7–34.6)</sub> | 0.09 | 78.7 <sub>(50.3–107.2)</sub> | 142.2 <sub>(69.8–214.6)</sub> | 52.4 | 93.1 | 16,595 | 1.0 | 50.5 | 1% | 169.6 | 0.0 | 0% | – |
| spec-ngram | 12/0 | 38.2 <sub>(38.0–38.3)</sub> | 0.15 | 58.6 <sub>(57.2–59.9)</sub> | 66.6 <sub>(66.1–67.2)</sub> | 27.1 | 42.7 | 7,983 | 1.0 | 30.8 | 1% | 293.1 | 0.0 | 0% | 29% |

**copy-c4**

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 | batch | step ms | peak KV | peak RSS MB | preempt | cached | draft accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spec-off | 24/0 | 67.2 <sub>(37.8–96.6)</sub> | 0.26 | 216.4 <sub>(201.5–231.3)</sub> | 312.3 <sub>(276.9–347.8)</sub> | 72.0 | 125.6 | 22,018 | 4.0 | 72.8 | 7% | 238.7 | 0.0 | 0% | – |
| spec-ngram | 24/0 | 91.3 <sub>(89.0–93.6)</sub> | 0.36 | 145.3 <sub>(128.6–162.0)</sub> | 217.0 <sub>(204.3–229.6)</sub> | 42.3 | 70.3 | 13,736 | 3.8 | 52.8 | 6% | 414.9 | 0.0 | 0% | 33% |

**copy-c16**

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 | batch | step ms | peak KV | peak RSS MB | preempt | cached |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spec-off | 48/0 | 147.2 <sub>(96.7–197.8)</sub> | 0.58 | 524.4 <sub>(366.7–682.1)</sub> | 744.4 <sub>(560.6–928.3)</sub> | 120.3 | 198.0 | 34,212 | 15.9 | 121.4 | 25% | 322.3 | 0.0 | 0% |
| spec-ngram | 48/0 | 147.4 <sub>(146.2–148.7)</sub> | 0.58 | 509.6 <sub>(485.7–533.4)</sub> | 804.2 <sub>(789.0–819.4)</sub> | 106.2 | 131.7 | 29,155 | 15.9 | 107.1 | 25% | 393.6 | 0.0 | 0% |
