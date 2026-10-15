"""Runtime configuration, read from ``MINISERVE_*`` environment variables (or a ``.env`` file)."""

from __future__ import annotations

import ipaddress
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
# Pinned so benchmark numbers are reproducible against the exact same weights.
DEFAULT_MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MINISERVE_", env_file=".env", extra="ignore")

    # --- backend / model ---
    # native: our paged-KV Llama engine | hf: Transformers DynamicCache baseline
    # toy: deterministic echo model, no torch | tiny: random 2-layer Llama on the native engine
    backend: Literal["native", "hf", "toy", "tiny"] = "native"
    model_id: str = DEFAULT_MODEL_ID
    model_revision: str | None = DEFAULT_MODEL_REVISION
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    max_model_len: int = Field(4096, ge=16)  # caps the context window (and the RoPE table)
    # Weight-only int8 (W8A16) for the native backend; KV cache and activations unchanged.
    quantization: Literal["none", "int8"] = "none"
    warmup: bool = True
    torch_threads: int | None = Field(None, ge=1)

    # --- KV cache + scheduler ---
    kv_cache_mb: float = Field(256.0, gt=0)
    block_size: int = Field(16, ge=1, le=256)
    max_num_seqs: int = Field(32, ge=1)
    max_num_batched_tokens: int = Field(512, ge=1)
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
    scheduling_policy: Literal["fcfs", "fair"] = "fcfs"

    # --- admission control / backpressure ---
    max_waiting: int = Field(256, ge=0)
    max_waiting_per_tenant: int = Field(64, ge=1)
    max_queued_tokens: int = Field(262_144, ge=1)

    # --- toy backend (simulated step cost, used for scheduling experiments) ---
    toy_prefill_ms_per_token: float = Field(0.0, ge=0)
    toy_decode_ms_per_token: float = Field(0.0, ge=0)
    toy_step_overhead_ms: float = Field(0.0, ge=0)
    toy_kv_blocks: int = Field(8192, ge=1)

    # --- network / safety ---
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)
    api_key: str | None = None
    # Multi-tenant keys "tenant:key,tenant2:key2"; the tenant is used for fair scheduling,
    # per-tenant queue limits and rate limiting.
    api_keys: str | None = None
    allow_unauthenticated_public: bool = False
    rate_limit_rpm: float = Field(600.0, ge=0)  # per tenant; 0 disables the limiter
    rate_limit_burst: int = Field(60, ge=1)

    # --- request limits ---
    max_prompt_chars: int = Field(65_536, ge=1)
    max_prompt_tokens: int = Field(4096, ge=1)
    max_new_tokens_cap: int = Field(1024, ge=1)
    default_max_tokens: int = Field(128, ge=1)
    request_timeout_s: float = Field(120.0, gt=0)

    # --- ops ---
    log_level: str = "info"
    log_format: Literal["text", "json"] = "text"
    drain_timeout_s: float = Field(30.0, ge=0)

    @field_validator("api_keys")
    @classmethod
    def _check_keys(cls, v: str | None) -> str | None:
        if v:
            for item in v.split(","):
                tenant, sep, key = item.strip().partition(":")
                if not sep or not tenant or not key:
                    raise ValueError("api_keys must look like 'tenant:key,tenant2:key2'")
        return v

    def key_to_tenant(self) -> dict[str, str]:
        keys = {}
        if self.api_keys:
            for item in self.api_keys.split(","):
                tenant, _, key = item.strip().partition(":")
                keys[key] = tenant
        if self.api_key:
            keys[self.api_key] = "default"
        return keys

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key or self.api_keys)

    def check_bind_safety(self) -> None:
        """Refuse to expose an unauthenticated, expensive endpoint beyond loopback."""
        if self.auth_enabled or self.allow_unauthenticated_public:
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
