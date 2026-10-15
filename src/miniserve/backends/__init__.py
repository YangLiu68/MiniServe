from __future__ import annotations

from miniserve.backends.base import Backend, SamplingParams
from miniserve.config import Settings

__all__ = ["Backend", "SamplingParams", "load_backend"]


def load_backend(settings: Settings) -> Backend:
    """Build the configured backend. Blocking (may download weights): call off the event loop."""
    if settings.backend == "toy":
        from miniserve.backends.toy import ToyBackend

        return ToyBackend(settings.toy_prefill_ms_per_token, settings.toy_decode_ms_per_token)
    # torch is an optional dependency: import only when a torch backend is requested.
    from miniserve.backends import hf

    if settings.backend == "tiny":
        return hf.build_tiny_random(device=hf.resolve_device(settings.device))
    return hf.HFBackend.from_pretrained(
        settings.model_id, settings.model_revision, settings.device, settings.dtype
    )
