"""Workload description: streams of requests with controlled prompt/output lengths and arrivals.

A *scenario* is a list of *streams* that run concurrently. Each stream belongs to one tenant and
generates requests either closed-loop (``concurrency`` workers, each sending its next request
when the previous finishes) or open-loop (Poisson arrivals at ``rate`` req/s, independent of
completions — the right model for latency-vs-load curves, since closed loops self-throttle).

Prompts are built to an exact token count with the *server's* tokenizer, from random words, and
every request gets a unique random prefix so that prefix caching only kicks in where a stream
explicitly asks for a ``shared_prefix_tokens`` system prompt. Alternatively a stream can use real
``prompts`` (a list, or ``"file.json:key"`` naming a list in a JSON file) — needed where content
matters, e.g. speculative decoding, whose payoff depends on how much the output copies the input.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Length = int | dict[str, list[int]]  # 128 or {"uniform": [64, 256]}


def sample_length(spec: Length, rng: random.Random) -> int:
    if isinstance(spec, int):
        return spec
    lo, hi = spec["uniform"]
    return rng.randint(lo, hi)


@dataclass
class Stream:
    tenant: str = "default"
    api_key: str | None = None
    mode: str = "closed"  # closed | poisson
    concurrency: int = 1
    rate: float = 1.0
    num_requests: int | None = None
    duration_s: float | None = None
    prompt_tokens: Length = 128
    max_tokens: Length = 128
    shared_prefix_tokens: int = 0
    warmup_s: float = 0.0  # requests that *start* before this are excluded from the summary
    temperature: float = 0.0
    endpoint: str = "completions"
    prompts: list[str] | str | None = None  # real prompts (cycled in a seeded shuffled order)
    # Fixed-length outputs (MiniServe/vLLM ignore EOS) make configs do identical work; turn off
    # where natural endings matter (e.g. speculative decoding: text after EOS is repetitive).
    ignore_eos: bool = True

    def __post_init__(self) -> None:
        if self.mode not in ("closed", "poisson"):
            raise ValueError(f"unknown stream mode {self.mode!r}")
        if self.num_requests is None and self.duration_s is None:
            raise ValueError("a stream needs num_requests or duration_s")
        if isinstance(self.prompts, str):
            path, _, key = self.prompts.partition(":")
            data = json.loads(Path(path).read_text())
            self.prompts = data[key] if key else data
        if self.prompts is not None and not (
            self.prompts and all(isinstance(p, str) and p for p in self.prompts)
        ):
            raise ValueError("prompts must be a non-empty list of non-empty strings")


@dataclass
class Scenario:
    name: str
    streams: list[Stream]
    description: str = ""
    seed: int = 0
    group: str | None = None  # scenarios sharing a group form one sweep curve in reports
    x: float | None = None  # this scenario's position on the sweep axis (e.g. concurrency)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Scenario:
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            seed=d.get("seed", 0),
            group=d.get("group"),
            x=d.get("x"),
            streams=[Stream(**s) for s in d["streams"]],
        )

    @classmethod
    def load(cls, path: str | Path) -> Scenario:
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass
class RequestSpec:
    stream: int
    tenant: str
    api_key: str | None
    prompt: str
    prompt_tokens: int  # as built client-side; the server's usage is the source of truth
    max_tokens: int
    temperature: float
    endpoint: str
    arrival: float | None = None  # seconds after start (open loop)
    extra: dict[str, Any] = field(default_factory=dict)
    ignore_eos: bool = True


class PromptFactory:
    """Builds prompts of an exact token length for a given tokenizer.

    ``bytes`` — byte tokenizer (toy/tiny backends): one ASCII char = one token.
    ``hf:<model_id>[@revision]`` — a Transformers tokenizer (MiniServe hf/native backends,
    and external servers serving the same model).
    """

    WORDS = (
        "time year people way day man thing woman life child world school state family student "
        "group country problem hand part place case week company system program question work "
        "government number night point home water room mother area money story fact month lot "
        "right study book eye job word business issue side kind head house service friend father "
        "power hour game line end member law car city community name president team minute idea "
        "kid body information back parent face others level office door health person art war "
        "history party result change morning reason research girl guy moment air teacher force "
        "education river garden music market light energy forest ocean bridge station window"
    ).split()

    def __init__(self, tokenizer: str = "bytes") -> None:
        self.kind = tokenizer
        self._tok = None
        if tokenizer.startswith("hf:"):
            from transformers import AutoTokenizer

            model, _, rev = tokenizer[3:].partition("@")
            self._tok = AutoTokenizer.from_pretrained(model, revision=rev or None)
        elif tokenizer != "bytes":
            raise ValueError("tokenizer must be 'bytes' or 'hf:<model_id>[@revision]'")

    def count(self, text: str) -> int:
        if self._tok is None:
            return len(text.encode())
        return len(self._tok(text, add_special_tokens=True)["input_ids"])

    def make(self, n_tokens: int, rng: random.Random, prefix: str = "") -> str:
        """Text whose token count (including ``prefix``) is exactly ``n_tokens`` when possible."""
        if self._tok is None:
            body_len = max(1, n_tokens - len(prefix.encode()))
            alphabet = "abcdefghijklmnopqrstuvwxyz "
            return prefix + "".join(rng.choice(alphabet) for _ in range(body_len))
        # Common words are ~1 token each: guess, then correct a few times (re-tokenizing after
        # every word would be quadratic for long prompts).
        head = f"{rng.randrange(10**6)}:"  # unique head: no accidental cross-request sharing
        words = [head] + [rng.choice(self.WORDS) for _ in range(max(0, n_tokens - 4))]
        text = prefix + " ".join(words)
        for _ in range(30):
            diff = n_tokens - self.count(text)
            if diff == 0:
                break
            if diff > 0:
                words += [rng.choice(self.WORDS) for _ in range(diff)]
            else:
                del words[max(1, len(words) + diff) :]
                if len(words) == 1:
                    break
            text = prefix + " ".join(words)
        return text


def build_requests(scenario: Scenario, factory: PromptFactory) -> list[list[RequestSpec]]:
    """Pre-build request specs per stream (deterministic for a given scenario seed).

    Duration-bounded streams get a generous pre-built pool; the runner stops at the deadline.
    """
    out = []
    for si, st in enumerate(scenario.streams):
        rng = random.Random(scenario.seed * 1000 + si)
        prefix = (
            factory.make(st.shared_prefix_tokens, rng) + "\n" if st.shared_prefix_tokens else ""
        )
        if st.num_requests is not None:
            n = st.num_requests
        elif st.mode == "poisson":
            n = int(st.rate * st.duration_s * 2 + 20)
        else:
            n = int(st.concurrency * st.duration_s * 4 + 20)  # >= 4 req/s per worker
        specs, t = [], 0.0
        order: list[str] = []
        for _ in range(n):
            arrival = None
            if st.mode == "poisson":
                t += rng.expovariate(st.rate)
                arrival = t
            if st.prompts is not None:
                if not order:  # each pass over the dataset in a new seeded order
                    order = rng.sample(st.prompts, len(st.prompts))
                prompt = prefix + order.pop()
            else:
                prompt = factory.make(sample_length(st.prompt_tokens, rng), rng, prefix)
            specs.append(
                RequestSpec(
                    si,
                    st.tenant,
                    st.api_key,
                    prompt,
                    factory.count(prompt),
                    sample_length(st.max_tokens, rng),
                    st.temperature,
                    st.endpoint,
                    arrival,
                    ignore_eos=st.ignore_eos,
                )
            )
        out.append(specs)
    return out
