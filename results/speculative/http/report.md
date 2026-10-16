### Suite `speculative`

n-gram (prompt lookup) speculative decoding vs plain decoding, SmolLM2-135M on MPS fp16, greedy, natural output lengths (no ignore_eos; up to 192 tokens) — the outputs are the same text in both configs, so latencies are directly comparable, and the number of decode steps is a contention-free measure of the effect. 'copy' prompts ask for edits/reformatting of a given text; 'open' prompts are free-form. Speculation is used only while <= 4 sequences decode, so with 16 clients it is off except at the tail of the run (a check that it does no harm under load). Prefix caching off to isolate the effect.

- Rounds: 2 (config order rotated each round; cells are the median across rounds, <sub>(min–max)</sub> where it varies)
- Host: Apple M4, 16.0 GB, macOS-15.3.2-arm64-arm-64bit; torch 2.14.1, transformers 5.19.0
- Commit: `069ae990c9` (dirty); load average at start [29.51, 20.32, 15.39], at end [9.52, 11.34, 12.64]
- Latencies in ms (client-side, streaming). TPOT = (E2E − TTFT)/(tokens − 1).

**copy-c1**

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 | batch | step ms | peak KV | peak RSS MB | preempt | cached | draft accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spec-off | 12/0 | 35.9 <sub>(35.8–35.9)</sub> | 0.42 | 57.8 <sub>(56.4–59.3)</sub> | 71.9 <sub>(66.0–77.7)</sub> | 27.0 | 33.2 | 5,341 | 1.0 | 27.3 | 1% | 698.4 | 0.0 | 0% | – |
| spec-ngram | 12/0 | 55.5 <sub>(54.7–56.3)</sub> | 0.65 | 60.4 <sub>(59.1–61.6)</sub> | 74.7 <sub>(67.3–82.1)</sub> | 17.4 | 40.6 | 3,649 | 1.0 | 31.7 | 1% | 751.1 | 0.0 | 0% | 51% |

**open-c1**

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 | batch | step ms | peak KV | peak RSS MB | preempt | cached | draft accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spec-off | 12/0 | 36.0 <sub>(35.0–36.9)</sub> | 0.20 | 49.2 <sub>(46.2–52.3)</sub> | 63.1 <sub>(62.1–64.2)</sub> | 27.2 | 33.5 | 5,755 | 1.0 | 27.6 | 1% | 699.2 | 0.0 | 0% | – |
| spec-ngram | 12/0 | 41.3 <sub>(41.2–41.5)</sub> | 0.23 | 47.5 <sub>(45.2–49.7)</sub> | 62.7 <sub>(60.0–65.5)</sub> | 25.6 | 35.6 | 5,167 | 1.0 | 27.8 | 1% | 747.5 | 0.0 | 0% | 29% |

**copy-c4**

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 | batch | step ms | peak KV | peak RSS MB | preempt | cached | draft accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spec-off | 24/0 | 91.0 <sub>(89.5–92.4)</sub> | 1.08 | 95.2 <sub>(94.0–96.5)</sub> | 150.5 <sub>(143.5–157.5)</sub> | 38.0 | 49.3 | 7,422 | 3.5 | 38.0 | 4% | 582.4 | 0.0 | 0% | – |
| spec-ngram | 24/0 | 149.9 <sub>(149.1–150.8)</sub> | 1.77 | 94.0 <sub>(93.8–94.2)</sub> | 114.8 <sub>(112.3–117.3)</sub> | 24.1 | 51.8 | 5,231 | 3.6 | 42.1 | 4% | 690.2 | 0.0 | 0% | 51% |

**copy-c16**

| config | ok/err | out tok/s | req/s | TTFT p50 | TTFT p95 | TPOT p50 | ITL p95 | E2E p95 | batch | step ms | peak KV | peak RSS MB | preempt | cached | draft accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spec-off | 48/0 | 192.7 <sub>(177.9–207.4)</sub> | 2.27 | 154.2 <sub>(148.3–160.1)</sub> | 533.4 <sub>(346.3–720.5)</sub> | 64.2 | 86.3 | 12,634 | 10.6 | 54.1 | 14% | 782.1 | 0.0 | 0% | – |
| spec-ngram | 48/0 | 186.4 <sub>(183.9–188.9)</sub> | 2.22 | 164.8 <sub>(164.4–165.1)</sub> | 367.0 <sub>(365.0–369.0)</sub> | 67.9 | 88.7 | 13,280 | 11.6 | 62.1 | 13% | 830.6 | 0.0 | 0% | 39% |
