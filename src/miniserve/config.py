"""Runtime configuration, read from ``MINISERVE_*`` environment variables (or a ``.env`` file)."""

from __future__ import annotations

import ipaddress
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
# Pinned so benchmark numbers are reproducible against the exact same weights.
DEFAULT_MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MINISERVE_", env_file=".env", extra="ignore")

    # --- backend / model ---
    backend: Literal["hf", "toy", "tiny"] = "hf"
    model_id: str = DEFAULT_MODEL_ID
    model_revision: str | None = DEFAULT_MODEL_REVISION
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    warmup: bool = True

    # --- toy backend (simulated compute cost, used for scheduling experiments) ---
    toy_prefill_ms_per_token: float = Field(0.0, ge=0)
    toy_decode_ms_per_token: float = Field(0.0, ge=0)

    # --- network / safety ---
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)
    api_key: str | None = None
    allow_unauthenticated_public: bool = False
    rate_limit_rpm: float = Field(120.0, ge=0)  # 0 disables the limiter
    rate_limit_burst: int = Field(20, ge=1)

    # --- request limits ---
    max_prompt_chars: int = Field(32_768, ge=1)
    max_prompt_tokens: int = Field(2048, ge=1)
    max_new_tokens_cap: int = Field(512, ge=1)
    default_max_tokens: int = Field(128, ge=1)
    request_timeout_s: float = Field(60.0, gt=0)
    max_pending: int = Field(8, ge=0)

    log_level: str = "info"

    def check_bind_safety(self) -> None:
        """Refuse to expose an unauthenticated, expensive endpoint beyond loopback."""
        if self.api_key or self.allow_unauthenticated_public:
            return
        if not _is_loopback(self.host):
            raise RuntimeError(
                f"Refusing to bind {self.host!r} without MINISERVE_API_KEY. Set an API key, bind "
                "127.0.0.1, or set MINISERVE_ALLOW_UNAUTHENTICATED_PUBLIC=1 if a trusted gateway "
                "in front of MiniServe handles auth."
            )


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
