# License notes

## This project

Not yet chosen — `pyproject.toml` says "TBD". The owner should pick one (MIT or Apache-2.0 are
both compatible with everything below) and add a `LICENSE` file before publishing.

## Model

| Model | License | Gated | Revision used | Verified |
|---|---|---|---|---|
| `HuggingFaceTB/SmolLM2-135M-Instruct` | Apache-2.0 | No | `12fd25f77366fa6b3b4b768ec3050bf629380bac` | HF Hub API `cardData.license`, 2026-10-14 |

Weights are downloaded at runtime into the Hugging Face cache and are **not** redistributed in
this repository. The `tiny` backend and all tests use randomly initialised weights generated
locally. Any other model set via `MINISERVE_MODEL_ID` carries its own license; check it before
use, especially for gated or non-commercial models.

## Python dependencies (installed versions, from package metadata, 2026-10-15)

| Package | Version | License | Used for |
|---|---|---|---|
| fastapi | 0.143.1 | MIT | HTTP API |
| starlette | 1.8.0 | BSD-3-Clause | (FastAPI) |
| uvicorn | 0.54.0 | BSD-3-Clause | ASGI server |
| pydantic / pydantic-settings | 2.14.1 / 2.16.0 | MIT | validation, config |
| prometheus-client | 0.26.0 | Apache-2.0 AND BSD-2-Clause | `/metrics` |
| psutil | 7.2.2 | BSD-3-Clause | RSS metric, benchmark env capture |
| torch (extra `hf`) | 2.14.1 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT | model execution |
| transformers (extra `hf`) | 5.19.0 | Apache-2.0 | checkpoint loading, tokenizer, reference model |
| tokenizers, safetensors, huggingface-hub | 0.23.3, 0.8.0, 2.2.0 | Apache-2.0 | (transformers) |
| numpy (transitive) | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | (torch/transformers) |
| httpx (extra `bench`, dev) | 0.28.1 | BSD-3-Clause | load generator, tests |
| matplotlib (extra `bench`) | 3.11.2 | Matplotlib License (PSF-style, BSD-compatible) | benchmark charts |
| pytest, ruff (dev) | 9.1.1, 0.17.0 | MIT | tests, lint |

No copyleft (GPL/AGPL) dependencies.

## Ideas and algorithms (reimplemented, not copied)

| Idea | Source | Where |
|---|---|---|
| Paged KV cache / block tables | Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention*, SOSP 2023 (vLLM, Apache-2.0) | `block_manager.py`, `backends/native.py` |
| Automatic prefix caching with chained block hashes | vLLM design docs | `block_manager.py` |
| Iteration-level (continuous) batching | Yu et al., *Orca*, OSDI 2022 | `scheduler.py` |
| Chunked prefill, decode-first ("stall-free") batching | Agrawal et al., *Sarathi-Serve*, OSDI 2024 | `scheduler.py` |
| Virtual Token Counter fair scheduling | Sheng et al., *Fairness in Serving LLMs*, OSDI 2024 | `policy.py` |
| Windowed incremental detokenization | Hugging Face TGI / vLLM (Apache-2.0) | `detokenizer.py` |
| Speculative decoding and its distribution-preserving acceptance rule | Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, ICML 2023; Chen et al., *Accelerating LLM Decoding with Speculative Sampling*, 2023 | `sampling.py`, `scheduler.py`, `backends/native.py` |
| N-gram / prompt-lookup drafts | Saxena, *Prompt Lookup Decoding* (2023); vLLM's `ngram` speculative method | `speculative.py` |
| Per-tenant prefix-cache isolation | vLLM's `cache_salt` | `sequence.py`, `engine.py` |
| Weight-only int8 with per-channel scales | common practice (e.g. gpt-fast, torchao) using PyTorch's `_weight_int8pack_mm` | `backends/native.py` |

The benchmark texts (`bench/data/eval_text.txt`, `bench/data/spec_prompts.json`) were written for
this repository.

## External tools (optional, not dependencies)

* **Ollama** (MIT) can be used as an external comparison target via the OpenAI-compatible client;
  its models carry their own licenses.
* **vLLM** (Apache-2.0) comparison requires a CUDA GPU; see BENCHMARK.md for the adapter.
