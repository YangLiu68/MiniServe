from __future__ import annotations

from miniserve.backends.base import Backend, SamplingParams
from miniserve.config import Settings

__all__ = ["Backend", "SamplingParams", "load_backend"]


def load_backend(settings: Settings) -> Backend:
    """Build the configured backend. Blocking (may download weights): call off the event loop."""
    if settings.backend == "toy":
        from miniserve.backends.toy import ToyBackend

        return ToyBackend(
            prefill_ms_per_token=settings.toy_prefill_ms_per_token,
            decode_ms_per_token=settings.toy_decode_ms_per_token,
            step_overhead_ms=settings.toy_step_overhead_ms,
            max_context=settings.max_model_len,
            num_kv_blocks=settings.toy_kv_blocks,
            block_size=settings.block_size,
        )
    # torch is an optional dependency: import only when a torch backend is requested.
    import torch

    if settings.torch_threads:
        torch.set_num_threads(settings.torch_threads)
    kv = dict(
        kv_cache_mb=settings.kv_cache_mb,
        block_size=settings.block_size,
        max_context=settings.max_model_len,
    )
    if settings.backend == "tiny":
        from miniserve.backends.hf import resolve_device
        from miniserve.backends.native import NativeBackend, build_tiny_random

        _, hf_model = build_tiny_random(device=resolve_device(settings.device))
        from miniserve.backends.toy import ByteTokenizer

        return NativeBackend.from_hf_model(hf_model, ByteTokenizer(), "tiny-random-llama", **kv)
    if settings.backend == "hf":
        from miniserve.backends.hf import HFBackend

        return HFBackend.from_pretrained(
            settings.model_id, settings.model_revision, settings.device, settings.dtype, **kv
        )
    from miniserve.backends.native import NativeBackend

    return NativeBackend.from_pretrained(
        settings.model_id, settings.model_revision, settings.device, settings.dtype, **kv
    )
