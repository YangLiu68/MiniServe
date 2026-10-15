"""Request models: a subset of the OpenAI Completions / Chat Completions APIs plus extensions.

Keeping the wire format OpenAI-compatible lets the same load generator target MiniServe and
mature servers (vLLM, llama.cpp, Ollama) for comparisons. Unknown fields are rejected (422)
rather than silently ignored, so a client never believes an unsupported option took effect.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_STOP_STRINGS = 4
MAX_STOP_CHARS = 64


class _GenerationParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    stop: str | list[str] | None = None
    stream: bool = False
    # Accepted for OpenAI-client compatibility; MiniServe always reports usage.
    stream_options: dict[str, bool] | None = None
    user: str | None = Field(default=None, max_length=256)
    # Extensions (vLLM-compatible name for ignore_eos): fixed-length outputs for benchmarking.
    ignore_eos: bool = False
    timeout_s: float | None = Field(default=None, gt=0)

    @field_validator("stop")
    @classmethod
    def _check_stop(cls, v: str | list[str] | None) -> str | list[str] | None:
        items = [v] if isinstance(v, str) else (v or [])
        if len(items) > MAX_STOP_STRINGS:
            raise ValueError(f"at most {MAX_STOP_STRINGS} stop strings")
        if any(not 1 <= len(s) <= MAX_STOP_CHARS for s in items):
            raise ValueError(f"stop strings must be 1..{MAX_STOP_CHARS} characters")
        return v

    @property
    def stop_tuple(self) -> tuple[str, ...]:
        if self.stop is None:
            return ()
        return (self.stop,) if isinstance(self.stop, str) else tuple(self.stop)


class CompletionRequest(_GenerationParams):
    prompt: str = Field(min_length=1)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(_GenerationParams):
    messages: list[ChatMessage] = Field(min_length=1, max_length=256)
    max_completion_tokens: int | None = Field(default=None, ge=1)  # newer OpenAI name
