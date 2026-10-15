"""Streaming OpenAI-compatible client that records per-request latency at chunk granularity.

Works against MiniServe and other OpenAI-compatible servers (vLLM, llama.cpp server, Ollama).
TTFT is measured to the first chunk carrying non-empty text (a chat "role" preamble does not
count). Inter-chunk gaps approximate inter-token latency; token counts come from the server's
``usage`` when it reports it (MiniServe always does; vLLM with ``stream_options``), otherwise
from the number of content chunks.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from miniserve.bench.workload import RequestSpec


@dataclass
class RequestResult:
    stream: int
    tenant: str
    start: float  # seconds since run start
    prompt_tokens: int
    output_tokens: int = 0
    cached_tokens: int = 0
    ttft: float | None = None
    e2e: float | None = None
    itl: list[float] = field(default_factory=list)
    finish_reason: str | None = None
    status: int | None = None
    error: str | None = None
    server: dict[str, Any] | None = None  # MiniServe's own timings, for cross-checking

    @property
    def ok(self) -> bool:
        return self.error is None and self.finish_reason in ("stop", "length")

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def build_payload(spec: RequestSpec, model: str | None, server_kind: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "max_tokens": spec.max_tokens,
        "temperature": spec.temperature,
        "stream": True,
    }
    if model:
        body["model"] = model
    if spec.endpoint == "chat":
        body["messages"] = [{"role": "user", "content": spec.prompt}]
    else:
        body["prompt"] = spec.prompt
    if server_kind in ("miniserve", "vllm"):
        body["ignore_eos"] = True  # fixed-length outputs: comparable token counts across runs
    if server_kind == "vllm":
        body["stream_options"] = {"include_usage": True}
    body.update(spec.extra)
    return body


async def send(
    client: httpx.AsyncClient,
    spec: RequestSpec,
    t0: float,
    model: str | None,
    server_kind: str = "miniserve",
) -> RequestResult:
    path = "/v1/chat/completions" if spec.endpoint == "chat" else "/v1/completions"
    headers = {"Authorization": f"Bearer {spec.api_key}"} if spec.api_key else {}
    start = time.perf_counter()
    res = RequestResult(spec.stream, spec.tenant, start - t0, spec.prompt_tokens)
    last = None
    chunks = 0
    try:
        async with client.stream(
            "POST", path, json=build_payload(spec, model, server_kind), headers=headers
        ) as r:
            res.status = r.status_code
            if r.status_code != 200:
                res.error = f"HTTP {r.status_code}: {(await r.aread())[:200]!r}"
                return res
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                now = time.perf_counter()
                choices = obj.get("choices") or []
                text = ""
                if choices:
                    c = choices[0]
                    text = c.get("text") or (c.get("delta") or {}).get("content") or ""
                    if c.get("finish_reason"):
                        res.finish_reason = c["finish_reason"]
                if text:
                    chunks += 1
                    if res.ttft is None:
                        res.ttft = now - start
                    elif last is not None:
                        res.itl.append(now - last)
                    last = now
                if usage := obj.get("usage"):
                    res.prompt_tokens = usage.get("prompt_tokens", res.prompt_tokens)
                    res.output_tokens = usage.get("completion_tokens", 0)
                    details = usage.get("prompt_tokens_details") or {}
                    res.cached_tokens = details.get("cached_tokens") or 0
                if ms := obj.get("miniserve"):
                    res.server = ms.get("timings")
                if err := obj.get("error"):
                    res.error = str(err)
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        res.error = f"{type(e).__name__}: {e}"
    res.e2e = time.perf_counter() - start
    if not res.output_tokens:
        res.output_tokens = chunks
    return res
