"""Request/response models: a subset of the OpenAI Completions API plus a ``miniserve`` extension.

Keeping the wire format OpenAI-compatible lets the same load generator target MiniServe and
mature servers (vLLM, llama.cpp server) for apples-to-apples comparisons later.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)
    model: str | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    stream: bool = False
    # MiniServe extension: per-request deadline in seconds, capped by the server maximum.
    timeout_s: float | None = Field(default=None, gt=0)
