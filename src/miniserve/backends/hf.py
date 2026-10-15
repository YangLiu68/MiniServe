"""Generic PyTorch + Transformers backend (the *integration* baseline).

Works with any ``AutoModelForCausalLM``. Each sequence keeps its own Transformers
``DynamicCache`` — this is integration with the library's KV cache, **not** an original cache
implementation (that is ``backends/native.py`` + ``block_manager.py``). A step runs each
scheduled sequence's span as a separate forward call, so the engine's continuous scheduling
still interleaves requests token by token, but there are no batched GEMMs and no prefix sharing.
It is the reference that the native backend is measured against at matched model/hardware.
"""

from __future__ import annotations

import gc
import logging

import torch

from miniserve.sampling import Sampler
from miniserve.sequence import ScheduledSeq, Sequence

log = logging.getLogger(__name__)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype != "auto":
        return getattr(torch, dtype)
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


class HFTokenizer:
    """Thin wrapper giving a Transformers tokenizer the backend ``Tokenizer`` interface."""

    def __init__(self, tok, extra_eos: frozenset[int] = frozenset()) -> None:
        self._tok = tok
        eos = {tok.eos_token_id} if tok.eos_token_id is not None else set()
        self.eos_token_ids = frozenset(eos | extra_eos)

    def encode(self, text: str) -> list[int]:
        return list(self._tok(text, add_special_tokens=True)["input_ids"])

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)

    def apply_chat_template(self, messages: list[dict[str, str]]) -> list[int]:
        if not getattr(self._tok, "chat_template", None):
            raise ValueError("this model's tokenizer has no chat template")
        text = self._tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return list(self._tok(text, add_special_tokens=False)["input_ids"])


def load_hf_model(model_id: str, revision: str | None, device: str, dtype: torch.dtype):
    """Load model + tokenizer; returns (model on device in eval mode, HFTokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("loading %s@%s on %s (%s)", model_id, revision or "main", device, dtype)
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype)
    model.to(device).eval()
    gen_eos = model.generation_config.eos_token_id
    gen_eos = gen_eos if isinstance(gen_eos, list) else [gen_eos] if gen_eos is not None else []
    return model, HFTokenizer(tok, frozenset(gen_eos))


def _kv_bytes_per_token(config, dtype: torch.dtype) -> int:
    heads = config.num_attention_heads
    kv_heads = getattr(config, "num_key_value_heads", None) or heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // heads
    elem = torch.empty((), dtype=dtype).element_size()
    return 2 * config.num_hidden_layers * kv_heads * head_dim * elem


class HFBackend:
    name = "hf"
    supports_prefix_caching = False

    def __init__(
        self,
        model,
        tokenizer,
        model_id: str,
        kv_cache_mb: float = 256.0,
        block_size: int = 16,
        max_context: int | None = None,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.model_id = model_id
        p = next(model.parameters())
        self.device = p.device.type
        self.dtype = str(p.dtype).removeprefix("torch.")
        cfg_ctx = int(getattr(model.config, "max_position_embeddings", 2048))
        self.max_context = min(cfg_ctx, max_context or cfg_ctx)
        self.block_size = block_size
        # DynamicCache memory is not paged; the block budget is an accounting device so the
        # scheduler's admission/preemption logic applies identically to both backends.
        per_token = _kv_bytes_per_token(model.config, p.dtype)
        self.num_kv_blocks = max(1, int(kv_cache_mb * 2**20) // (per_token * block_size))
        self._caches: dict[str, object] = {}
        self.sampler = Sampler()

    @classmethod
    def from_pretrained(
        cls,
        model_id,
        revision,
        device="auto",
        dtype="auto",
        kv_cache_mb=256.0,
        block_size=16,
        max_context=None,
    ) -> HFBackend:
        dev = resolve_device(device)
        model, tok = load_hf_model(model_id, revision, dev, resolve_dtype(dtype, dev))
        return cls(model, tok, model_id, kv_cache_mb, block_size, max_context)

    @torch.inference_mode()
    def execute(self, batch: list[ScheduledSeq]) -> list[int | None]:
        rows, sampling = [], []
        for s in batch:
            rid = s.seq.request_id
            cache = self._caches.get(rid) if s.start > 0 else None
            ids = torch.tensor([s.token_ids], dtype=torch.long, device=self.device)
            out = self.model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
            self._caches[rid] = out.past_key_values
            if s.sample:
                rows.append(out.logits[0, -1])
                sampling.append(s.seq)
        tokens = iter(self.sampler.sample(torch.stack(rows), sampling) if rows else [])
        return [next(tokens) if s.sample else None for s in batch]

    def release(self, seq: Sequence, preempted: bool = False) -> None:
        self._caches.pop(seq.request_id, None)
        self.sampler.release(seq, preempted)

    def memory_stats(self) -> dict[str, float]:
        from miniserve.backends.native import device_memory

        return device_memory(self.device)

    def close(self) -> None:
        from miniserve.backends.native import empty_device_cache

        self.model = None
        self._caches.clear()
        gc.collect()
        empty_device_cache(self.device)
