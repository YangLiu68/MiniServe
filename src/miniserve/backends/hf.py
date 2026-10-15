"""PyTorch + Transformers backend.

Generation is a hand-written prefill + decode loop (not ``model.generate``) so the engine can
stop between any two tokens. The KV cache is Transformers' own ``DynamicCache``, passed back in
via ``past_key_values`` — that is *integration* with the library cache, not an original cache
implementation. Sampling (greedy / temperature / top-p, per-request seeded ``torch.Generator``)
runs on CPU over the last-position logits, so results with a fixed seed do not depend on device
RNG behaviour (they can still differ across devices/dtypes through the logits themselves).
"""

from __future__ import annotations

import gc
import inspect
import logging
from collections.abc import Iterator

import torch

from miniserve.backends.base import SamplingParams

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


def sample_next(logits: torch.Tensor, params: SamplingParams, generator: torch.Generator) -> int:
    """Pick the next token from 1-D float32 CPU logits."""
    if params.temperature == 0:
        return int(torch.argmax(logits))
    probs = torch.softmax(logits / params.temperature, dim=-1)
    if params.top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        mass_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
        sorted_probs[mass_before > params.top_p] = 0.0  # always keeps the top token
        choice = torch.multinomial(sorted_probs, 1, generator=generator)
        return int(sorted_idx[choice])
    return int(torch.multinomial(probs, 1, generator=generator))


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


class HFBackend:
    name = "hf"

    def __init__(self, model, tokenizer, model_id: str, device: str) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.device = device
        self.dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
        self.max_context = int(getattr(model.config, "max_position_embeddings", 2048))
        # Prefill only needs the last position's logits; avoids a [prompt_len, vocab] tensor.
        self._logits_kw = (
            {"logits_to_keep": 1}
            if "logits_to_keep" in inspect.signature(model.forward).parameters
            else {}
        )

    @classmethod
    def from_pretrained(
        cls, model_id: str, revision: str | None, device: str = "auto", dtype: str = "auto"
    ) -> HFBackend:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dev = resolve_device(device)
        torch_dtype = resolve_dtype(dtype, dev)
        log.info("loading %s@%s on %s (%s)", model_id, revision or "main", dev, torch_dtype)
        tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch_dtype)
        model.to(dev)
        gen_eos = model.generation_config.eos_token_id
        gen_eos = gen_eos if isinstance(gen_eos, list) else [gen_eos] if gen_eos is not None else []
        return cls(model, HFTokenizer(tok, frozenset(gen_eos)), model_id, dev)

    @torch.inference_mode()
    def generate(self, prompt_ids: list[int], params: SamplingParams) -> Iterator[int]:
        gen = torch.Generator(device="cpu")
        if params.seed is not None:
            gen.manual_seed(params.seed)
        else:
            gen.seed()
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        out = self.model(input_ids=input_ids, use_cache=True, **self._logits_kw)
        while True:
            tok = sample_next(out.logits[0, -1].float().cpu(), params, gen)
            yield tok
            out = self.model(
                input_ids=torch.tensor([[tok]], dtype=torch.long, device=self.device),
                past_key_values=out.past_key_values,
                use_cache=True,
                **self._logits_kw,
            )

    def close(self) -> None:
        self.model = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()


def build_tiny_random(device: str = "cpu", seed: int = 0) -> HFBackend:
    """A 2-layer randomly initialised GPT-2 over the byte tokenizer — no download needed.

    Its outputs are gibberish, but it runs the real PyTorch/Transformers code path (prefill,
    DynamicCache decode, sampling), which is what the tests need.
    """
    from transformers import GPT2Config, GPT2LMHeadModel

    from miniserve.backends.toy import ByteTokenizer

    torch.manual_seed(seed)
    cfg = GPT2Config(
        vocab_size=ByteTokenizer.vocab_size,
        n_positions=1024,
        n_embd=64,
        n_layer=2,
        n_head=2,
        bos_token_id=ByteTokenizer.EOS,
        eos_token_id=ByteTokenizer.EOS,
    )
    model = GPT2LMHeadModel(cfg).to(device)
    return HFBackend(model, ByteTokenizer(), model_id="tiny-random-gpt2", device=device)
