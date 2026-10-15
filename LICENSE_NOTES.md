# License notes

## This project

Not yet chosen — `pyproject.toml` says "TBD". The owner should pick one (MIT or Apache-2.0 are
both compatible with everything below) and add a `LICENSE` file before publishing.

## Model

| Model | License | Gated | Revision used | Verified |
|---|---|---|---|---|
| `HuggingFaceTB/SmolLM2-135M-Instruct` | Apache-2.0 | No | `12fd25f77366fa6b3b4b768ec3050bf629380bac` | HF Hub API `cardData.license`, 2026-10-14 |

Weights are downloaded at runtime into the Hugging Face cache and are **not** redistributed in
this repository. The `tiny` backend uses randomly initialised weights generated locally (no
third-party weights). Any other model set via `MINISERVE_MODEL_ID` carries its own license;
check it before use, especially for gated or non-commercial models.

## Python dependencies (installed versions, from package metadata, 2026-10-14)

| Package | Version | License |
|---|---|---|
| fastapi | 0.143.1 | MIT |
| starlette | 1.8.0 | BSD-3-Clause |
| uvicorn | 0.54.0 | BSD-3-Clause |
| pydantic | 2.14.1 | MIT |
| pydantic-settings | 2.16.0 | MIT |
| torch (optional `hf` extra) | 2.14.1 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT (package metadata expression) |
| transformers (optional) | 5.19.0 | Apache-2.0 |
| tokenizers (optional) | 0.23.3 | Apache-2.0 |
| safetensors (optional) | 0.8.0 | Apache-2.0 |
| huggingface-hub (optional) | 2.2.0 | Apache-2.0 |
| numpy (transitive) | 2.5.4 | BSD-3-Clause and others |
| httpx (dev) | 0.28.1 | BSD-3-Clause |
| pytest (dev) | 9.1.1 | MIT |
| ruff (dev) | 0.17.0 | MIT |

No copyleft (GPL/AGPL) dependencies. The incremental detokenization scheme follows the approach
used in Hugging Face TGI and vLLM (both Apache-2.0); it is reimplemented here, not copied.
