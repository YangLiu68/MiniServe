"""Native paged-KV Llama-family model with ragged continuous batching (original implementation).

Weights are taken from the Hugging Face model (so loading/tokenization/architecture configs are
the library's), but the forward pass, the KV cache and the batching are ours:

* **Paged KV cache** — one tensor per layer of ``num_blocks * block_size`` token slots. A sequence
  owns a block table (``Sequence.block_ids``, managed by ``BlockManager``); token at position
  ``p`` lives in slot ``block_ids[p // block_size] * block_size + p % block_size``. Blocks can be
  shared between sequences (prefix caching) because they are addressed indirectly.
* **Ragged batching** — a step mixes prefill chunks of different lengths and single-token decodes.
  Projections/MLP run on the flattened ``[total_tokens, hidden]`` tensor (no padding waste);
  only attention pads, to ``[batch, max_query_len]`` queries x ``[batch, max_context]`` keys.
* **Paged attention, PyTorch edition** — keys/values are *gathered* from the block tables into a
  padded tensor and fed to ``scaled_dot_product_attention`` with a boolean mask (causal within
  the chunk, full over the cached context, nothing beyond a sequence's length). A real
  PagedAttention kernel reads blocks in place instead of materialising the gather; that is the
  main performance gap to vLLM and is documented as such.
* Fused QKV and gate/up projections (one GEMM each instead of three/two).

Correctness is pinned by tests against the Transformers implementation: logits for whole
prompts, chunked prefill, mixed batches and prefix-cache hits match HF within float tolerance.

Supported: LlamaForCausalLM, Qwen2ForCausalLM (QKV bias), MistralForCausalLM without an active
sliding window; RoPE variants whose frequencies do not depend on sequence length.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from miniserve.backends.hf import HFTokenizer, load_hf_model, resolve_device, resolve_dtype
from miniserve.sampling import Sampler
from miniserve.sequence import ScheduledSeq, Sequence

log = logging.getLogger(__name__)

SUPPORTED_ARCHITECTURES = ("LlamaForCausalLM", "Qwen2ForCausalLM", "MistralForCausalLM")


class UnsupportedModel(ValueError):
    pass


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # Same op order/precision as transformers' LlamaRMSNorm (normalise in fp32, cast back).
    dtype = x.dtype
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x32.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@dataclass
class Int8Weight:
    """Weight-only int8 quantization (W8A16): symmetric, one scale per output channel.

    ``q`` is int8 [out, in]; ``scale`` is [out] in the compute dtype; W ~= q * scale[:, None].
    Activations stay in fp16/bf16/fp32. Per-output-channel scales make the matmul
    ``(x @ q.T) * scale``, which PyTorch fuses in ``torch._weight_int8pack_mm`` on CPU/MPS.
    """

    q: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def from_float(cls, w: torch.Tensor) -> Int8Weight:
        w32 = w.float()
        scale = w32.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        q = torch.round(w32 / scale[:, None]).clamp(-127, 127).to(torch.int8)
        return cls(q.contiguous(), scale.to(w.dtype))

    @property
    def nbytes(self) -> int:
        return self.q.numel() + self.scale.numel() * self.scale.element_size()

    def dequantize_rows(self, rows: torch.Tensor) -> torch.Tensor:
        return self.q[rows].to(self.scale.dtype) * self.scale[rows, None]


_INT8_MM_OK: dict[tuple[str, torch.dtype], bool] = {}


def _int8_mm_supported(device: torch.device, dtype: torch.dtype) -> bool:
    key = (device.type, dtype)
    if key not in _INT8_MM_OK:
        try:
            x = torch.ones(2, 32, device=device, dtype=dtype)
            w = torch.ones(8, 32, device=device, dtype=torch.int8)
            torch._weight_int8pack_mm(x, w, torch.ones(8, device=device, dtype=dtype))
            _INT8_MM_OK[key] = True
        except Exception:  # pragma: no cover - depends on torch build/device
            _INT8_MM_OK[key] = False
    return _INT8_MM_OK[key]


def linear(x: torch.Tensor, w, b: torch.Tensor | None = None) -> torch.Tensor:
    if isinstance(w, Int8Weight):
        if _int8_mm_supported(x.device, x.dtype):
            y = torch._weight_int8pack_mm(x, w.q, w.scale)
        else:  # dequantize-on-the-fly fallback
            y = F.linear(x, w.q.to(x.dtype)) * w.scale
        return y if b is None else y + b
    return F.linear(x, w, b)


@dataclass
class Layer:
    ln1: torch.Tensor
    ln2: torch.Tensor
    wqkv: torch.Tensor
    bqkv: torch.Tensor | None
    wo: torch.Tensor
    bo: torch.Tensor | None
    wgu: torch.Tensor
    bgu: torch.Tensor | None
    wdown: torch.Tensor
    bdown: torch.Tensor | None


def _cat_bias(*linears) -> torch.Tensor | None:
    biases = [lin.bias for lin in linears]
    if all(b is None for b in biases):
        return None
    return torch.cat(
        [
            b if b is not None else torch.zeros_like(lin.weight[:, 0])
            for b, lin in zip(biases, linears, strict=True)
        ]
    )


@dataclass
class StepMetadata:
    """Everything a step needs besides the weights; built once per step, shared by all layers."""

    input_ids: torch.Tensor  # [T]
    positions: torch.Tensor  # [T]
    slot_mapping: torch.Tensor  # [T]   cache slot each new token's K/V is written to
    gather_slots: torch.Tensor  # [B, L] cache slots to read for each sequence (0 = padding)
    attn_mask: torch.Tensor  # [B, 1, Q, L] bool, True = may attend
    q_index: torch.Tensor  # [T]   flat row in the padded [B*Q] query layout
    sample_index: torch.Tensor  # [S]   rows of T whose hidden state produces logits
    batch: int
    max_q: int
    max_ctx: int


def build_metadata(batch: list[ScheduledSeq], block_size: int, device: str) -> StepMetadata:
    B = len(batch)
    q_lens = [s.num_tokens for s in batch]
    starts = [s.start for s in batch]
    Q = max(q_lens)
    ctx_lens = [st + q for st, q in zip(starts, q_lens, strict=True)]
    L = max(ctx_lens)
    max_blocks = max(len(s.seq.block_ids) for s in batch)

    ids: list[int] = []
    pos: list[int] = []
    q_index: list[int] = []
    tok_seq: list[int] = []
    sample_index: list[int] = []
    tables = torch.zeros(B, max_blocks, dtype=torch.long)
    for b, s in enumerate(batch):
        n = s.num_tokens
        ids += s.token_ids
        pos += range(s.start, s.start + n)
        q_index += range(b * Q, b * Q + n)
        tok_seq += [b] * n
        tables[b, : len(s.seq.block_ids)] = torch.tensor(s.seq.block_ids, dtype=torch.long)
        if s.sample:
            sample_index.append(len(ids) - 1)

    pos_t = torch.tensor(pos, dtype=torch.long)
    slot_mapping = (
        tables[torch.tensor(tok_seq), pos_t // block_size] * block_size + pos_t % block_size
    )

    j = torch.arange(L)
    ctx_t = torch.tensor(ctx_lens)
    key_valid = j[None, :] < ctx_t[:, None]  # [B, L]
    gather = tables[:, (j // block_size).clamp(max=max_blocks - 1)] * block_size + j % block_size
    gather = torch.where(key_valid, gather, torch.zeros_like(gather))
    qi = torch.arange(Q)
    q_pos = torch.tensor(starts)[:, None] + qi[None, :]  # [B, Q] absolute positions
    q_valid = qi[None, :] < torch.tensor(q_lens)[:, None]
    mask = (j[None, None, :] <= q_pos[:, :, None]) & key_valid[:, None, :]
    # padded query rows attend to key 0 only (an all-False row would make softmax produce NaN)
    mask |= ~q_valid[:, :, None] & (j == 0)[None, None, :]

    def dev(t: torch.Tensor) -> torch.Tensor:
        return t.to(device, non_blocking=True)

    return StepMetadata(
        input_ids=dev(torch.tensor(ids, dtype=torch.long)),
        positions=dev(pos_t),
        slot_mapping=dev(slot_mapping),
        gather_slots=dev(gather),
        attn_mask=dev(mask[:, None]),
        q_index=dev(torch.tensor(q_index, dtype=torch.long)),
        sample_index=dev(torch.tensor(sample_index, dtype=torch.long)),
        batch=B,
        max_q=Q,
        max_ctx=L,
    )


class PagedKVCache:
    def __init__(self, num_layers, num_blocks, block_size, num_kv_heads, head_dim, dtype, device):
        shape = (num_layers, num_blocks * block_size, num_kv_heads, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)

    @property
    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2

    @staticmethod
    def bytes_per_token(num_layers, num_kv_heads, head_dim, dtype) -> int:
        return (
            2 * num_layers * num_kv_heads * head_dim * torch.empty((), dtype=dtype).element_size()
        )


class NativeLlama:
    def __init__(self, hf_model, max_context: int | None = None) -> None:
        cfg = hf_model.config
        arch = (cfg.architectures or [type(hf_model).__name__])[0]
        if arch not in SUPPORTED_ARCHITECTURES:
            raise UnsupportedModel(
                f"{arch} is not supported (supported: {SUPPORTED_ARCHITECTURES})"
            )
        window = getattr(cfg, "sliding_window", None)
        if (
            window
            and getattr(cfg, "use_sliding_window", True)
            and window < cfg.max_position_embeddings
        ):
            raise UnsupportedModel(f"sliding-window attention ({window}) is not implemented")
        rotary = hf_model.model.rotary_emb
        if "dynamic" in str(getattr(rotary, "rope_type", "default")):
            raise UnsupportedModel("dynamic RoPE scaling is not implemented")

        self.arch = arch
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", None) or self.num_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.num_heads
        self.eps = cfg.rms_norm_eps
        self.num_layers = cfg.num_hidden_layers
        self.max_context = min(
            cfg.max_position_embeddings, max_context or cfg.max_position_embeddings
        )
        self.vocab_size = cfg.vocab_size

        m = hf_model.model
        p = next(hf_model.parameters())
        self.dtype, self.device = p.dtype, p.device
        self.embed = m.embed_tokens.weight
        self.norm = m.norm.weight
        self.lm_head = hf_model.lm_head.weight  # tied to embed for SmolLM2
        self.layers: list[Layer] = []
        for layer in m.layers:
            a, mlp = layer.self_attn, layer.mlp
            self.layers.append(
                Layer(
                    ln1=layer.input_layernorm.weight,
                    ln2=layer.post_attention_layernorm.weight,
                    wqkv=torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]),
                    bqkv=_cat_bias(a.q_proj, a.k_proj, a.v_proj),
                    wo=a.o_proj.weight,
                    bo=a.o_proj.bias,
                    wgu=torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight]),
                    bgu=_cat_bias(mlp.gate_proj, mlp.up_proj),
                    wdown=mlp.down_proj.weight,
                    bdown=mlp.down_proj.bias,
                )
            )
        # RoPE table, computed like transformers' rotary embedding (fp32 product, then cast).
        inv_freq = rotary.inv_freq.detach().float().cpu()
        positions = torch.arange(self.max_context, dtype=torch.float32)
        freqs = positions[:, None] * inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        scale = float(getattr(rotary, "attention_scaling", 1.0))
        self.cos = (emb.cos() * scale).to(device=self.device, dtype=self.dtype)
        self.sin = (emb.sin() * scale).to(device=self.device, dtype=self.dtype)
        self.quantization = "none"
        self._probe_gqa()

    def _probe_gqa(self) -> None:
        """Use SDPA's native GQA if this torch/device supports it, else repeat K/V heads."""
        self._sdpa_gqa = False
        if self.num_heads == self.num_kv_heads:
            return
        try:
            q = torch.zeros(1, self.num_heads, 1, 8, dtype=self.dtype, device=self.device)
            kv = torch.zeros(1, self.num_kv_heads, 2, 8, dtype=self.dtype, device=self.device)
            mask = torch.ones(1, 1, 1, 2, dtype=torch.bool, device=self.device)
            F.scaled_dot_product_attention(q, kv, kv, attn_mask=mask, enable_gqa=True)
            self._sdpa_gqa = True
        except Exception:  # pragma: no cover - depends on torch build
            log.info("SDPA enable_gqa unavailable on %s; repeating KV heads", self.device)

    def weight_bytes(self) -> int:
        ts = [self.embed, self.norm] + ([] if self.lm_head is self.embed else [self.lm_head])
        for L in self.layers:
            ts += [t for t in vars(L).values() if t is not None]
        seen, total = set(), 0
        for t in ts:
            if id(t) in seen:
                continue
            seen.add(id(t))
            total += t.nbytes if isinstance(t, Int8Weight) else t.numel() * t.element_size()
        return total

    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        if isinstance(self.embed, Int8Weight):
            return self.embed.dequantize_rows(ids)
        return F.embedding(ids, self.embed)

    def quantize_int8(self) -> None:
        """Replace every projection (and the LM head / embedding) with int8 weights.

        Norm weights, biases and the KV cache stay in the compute dtype. A tied LM head and
        embedding share one int8 tensor: per-row scales serve both the matmul (per output
        channel = per vocab row) and the embedding lookup (dequantize the looked-up rows).
        """
        tied = self.lm_head is self.embed
        for L in self.layers:
            for name in ("wqkv", "wo", "wgu", "wdown"):
                setattr(L, name, Int8Weight.from_float(getattr(L, name)))
        self.lm_head = Int8Weight.from_float(self.lm_head)
        self.embed = self.lm_head if tied else Int8Weight.from_float(self.embed)
        self.quantization = "int8"

    def attention(self, q, k_cache, v_cache, meta: StepMetadata) -> torch.Tensor:
        T, H, D = q.shape
        B, Q = meta.batch, meta.max_q
        if Q == 1:  # pure decode step: one query per sequence, no padding needed
            qp = q.view(B, 1, H, D)
        else:
            qp = q.new_zeros(B * Q, H, D)
            qp[meta.q_index] = q
            qp = qp.view(B, Q, H, D)
        qp = qp.transpose(1, 2)  # [B, H, Q, D]
        kp = k_cache[meta.gather_slots].transpose(1, 2)  # [B, KVH, L, D]
        vp = v_cache[meta.gather_slots].transpose(1, 2)
        if self._sdpa_gqa:
            o = F.scaled_dot_product_attention(
                qp, kp, vp, attn_mask=meta.attn_mask, enable_gqa=True
            )
        else:
            rep = H // self.num_kv_heads
            if rep > 1:
                kp, vp = kp.repeat_interleave(rep, dim=1), vp.repeat_interleave(rep, dim=1)
            o = F.scaled_dot_product_attention(qp, kp, vp, attn_mask=meta.attn_mask)
        o = o.transpose(1, 2).reshape(B * Q, H, D)
        return o if Q == 1 else o[meta.q_index]

    @torch.inference_mode()
    def forward(self, meta: StepMetadata, kv: PagedKVCache) -> torch.Tensor:
        """Run one ragged step; returns logits [S, vocab] for ``meta.sample_index`` rows."""
        H, KVH, D = self.num_heads, self.num_kv_heads, self.head_dim
        h = self.embed_tokens(meta.input_ids)
        T = h.shape[0]
        cos = self.cos[meta.positions][:, None, :]
        sin = self.sin[meta.positions][:, None, :]
        for li, layer in enumerate(self.layers):
            x = rms_norm(h, layer.ln1, self.eps)
            q, k, v = linear(x, layer.wqkv, layer.bqkv).split([H * D, KVH * D, KVH * D], dim=-1)
            q = q.reshape(T, H, D)
            k = k.reshape(T, KVH, D)
            v = v.reshape(T, KVH, D)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            kv.k[li].index_copy_(0, meta.slot_mapping, k)
            kv.v[li].index_copy_(0, meta.slot_mapping, v)
            attn = self.attention(q, kv.k[li], kv.v[li], meta)
            h = h + linear(attn.reshape(T, H * D), layer.wo, layer.bo)
            x = rms_norm(h, layer.ln2, self.eps)
            gate, up = linear(x, layer.wgu, layer.bgu).chunk(2, dim=-1)
            h = h + linear(F.silu(gate) * up, layer.wdown, layer.bdown)
        hs = rms_norm(h[meta.sample_index], self.norm, self.eps)
        return linear(hs, self.lm_head)


class NativeBackend:
    name = "native"
    supports_prefix_caching = True

    def __init__(
        self,
        model: NativeLlama,
        tokenizer,
        model_id: str,
        num_kv_blocks: int,
        block_size: int = 16,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.device = str(model.device.type)
        self.dtype = str(model.dtype).removeprefix("torch.")
        if model.quantization != "none":
            self.dtype += f"+w{model.quantization}"
        self.max_context = model.max_context
        self.block_size = block_size
        self.num_kv_blocks = num_kv_blocks
        self.kv = PagedKVCache(
            model.num_layers,
            num_kv_blocks,
            block_size,
            model.num_kv_heads,
            model.head_dim,
            model.dtype,
            model.device,
        )
        self.sampler = Sampler()

    @staticmethod
    def blocks_for_budget(model: NativeLlama, kv_cache_mb: float, block_size: int) -> int:
        per_token = PagedKVCache.bytes_per_token(
            model.num_layers, model.num_kv_heads, model.head_dim, model.dtype
        )
        return max(1, int(kv_cache_mb * 2**20) // (per_token * block_size))

    @classmethod
    def from_hf_model(
        cls,
        hf_model,
        tokenizer,
        model_id,
        kv_cache_mb=256.0,
        block_size=16,
        max_context=None,
        quantization: str = "none",
    ) -> NativeBackend:
        model = NativeLlama(hf_model, max_context)
        if quantization == "int8":
            model.quantize_int8()
        elif quantization != "none":
            raise ValueError(f"unknown quantization {quantization!r}")
        blocks = cls.blocks_for_budget(model, kv_cache_mb, block_size)
        return cls(model, tokenizer, model_id, blocks, block_size)

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
        quantization="none",
    ) -> NativeBackend:
        dev = resolve_device(device)
        hf_model, tok = load_hf_model(model_id, revision, dev, resolve_dtype(dtype, dev))
        backend = cls.from_hf_model(
            hf_model, tok, model_id, kv_cache_mb, block_size, max_context, quantization
        )
        del hf_model  # our fused copies replace the per-projection weights
        gc.collect()
        return backend

    def execute(self, batch: list[ScheduledSeq]) -> list[int | None]:
        meta = build_metadata(batch, self.block_size, self.device)
        logits = self.model.forward(meta, self.kv)
        sampling = [s.seq for s in batch if s.sample]
        tokens = iter(self.sampler.sample(logits, sampling) if sampling else [])
        return [next(tokens) if s.sample else None for s in batch]

    def release(self, seq: Sequence, preempted: bool = False) -> None:
        self.sampler.release(seq, preempted)

    def memory_stats(self) -> dict[str, float]:
        stats = {"kv_cache_bytes": self.kv.nbytes, "weights_bytes": self.model.weight_bytes()}
        stats.update(device_memory(self.device))
        return stats

    def close(self) -> None:
        self.model = None
        self.kv = None
        gc.collect()
        empty_device_cache(self.device)


def device_memory(device: str) -> dict[str, float]:
    if device == "cuda":
        return {
            "device_allocated_bytes": torch.cuda.memory_allocated(),
            "device_peak_bytes": torch.cuda.max_memory_allocated(),
        }
    if device == "mps":
        return {
            "device_allocated_bytes": torch.mps.current_allocated_memory(),
            "device_driver_bytes": torch.mps.driver_allocated_memory(),
        }
    return {}


def empty_device_cache(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def build_tiny_random(
    device: str = "cpu",
    arch: str = "llama",
    seed: int = 0,
    kv_cache_mb=8.0,
    block_size: int = 16,
    dtype: torch.dtype = torch.float32,
):
    """Tiny randomly initialised Llama/Qwen2/Mistral over the byte tokenizer; returns
    (NativeBackend, hf_model) so tests can compare against the Transformers forward."""
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        MistralConfig,
        MistralForCausalLM,
        Qwen2Config,
        Qwen2ForCausalLM,
    )

    from miniserve.backends.toy import ByteTokenizer

    common = dict(
        vocab_size=ByteTokenizer.vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        rms_norm_eps=1e-6,
        bos_token_id=256,
        eos_token_id=256,
        tie_word_embeddings=False,
    )
    cfg_cls, model_cls = {
        "llama": (LlamaConfig, LlamaForCausalLM),
        "qwen2": (Qwen2Config, Qwen2ForCausalLM),
        "mistral": (MistralConfig, MistralForCausalLM),
    }[arch]
    extra = {"sliding_window": None} if arch == "mistral" else {}
    torch.manual_seed(seed)
    hf_model = model_cls(cfg_cls(**common, **extra)).to(device=device, dtype=dtype).eval()
    hf_model.config.architectures = [model_cls.__name__]
    backend = NativeBackend.from_hf_model(
        hf_model, ByteTokenizer(), f"tiny-random-{arch}", kv_cache_mb, block_size
    )
    return backend, hf_model


__all__ = ["NativeBackend", "NativeLlama", "HFTokenizer", "build_tiny_random", "UnsupportedModel"]
