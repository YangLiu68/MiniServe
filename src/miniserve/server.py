"""FastAPI app: lifecycle, auth + tenancy, limits, completions / chat completions (JSON + SSE),
cancellation, health, readiness, metrics."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import math
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from miniserve import __version__
from miniserve.backends import Backend, SamplingParams, load_backend
from miniserve.config import Settings
from miniserve.engine import (
    Engine,
    EngineClosed,
    EngineConfig,
    EngineOverloaded,
    FinishEvent,
    FinishReason,
    RequestHandle,
    TokenEvent,
)
from miniserve.metrics import Metrics
from miniserve.middleware import BodySizeLimitMiddleware
from miniserve.ratelimit import RateLimiter
from miniserve.schemas import ChatCompletionRequest, CompletionRequest

log = logging.getLogger("miniserve")
Kind = Literal["text", "chat"]


class APIError(Exception):
    def __init__(
        self,
        status: int,
        message: str,
        type_: str = "invalid_request_error",
        code: str | None = None,
        headers: dict[str, str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status, self.message, self.type, self.code = status, message, type_, code
        self.headers, self.extra = headers, extra or {}

    def response(self) -> JSONResponse:
        err = {"message": self.message, "type": self.type, "code": self.code}
        return JSONResponse({"error": err, **self.extra}, self.status, headers=self.headers)


def engine_config(s: Settings) -> EngineConfig:
    return EngineConfig(
        max_num_seqs=s.max_num_seqs,
        max_num_batched_tokens=s.max_num_batched_tokens,
        enable_chunked_prefill=s.enable_chunked_prefill,
        enable_prefix_caching=s.enable_prefix_caching,
        policy=s.scheduling_policy,
        max_waiting=s.max_waiting,
        max_waiting_per_tenant=s.max_waiting_per_tenant,
        max_queued_tokens=s.max_queued_tokens,
        speculative=s.speculative,
        num_speculative_tokens=s.num_speculative_tokens,
        ngram_max=s.spec_ngram_max,
        ngram_min=s.spec_ngram_min,
        spec_max_batch=s.spec_max_batch,
        prefix_cache_scope=s.prefix_cache_scope,
    )


async def _warmup(engine: Engine) -> None:
    """A few short concurrent requests through the real batched path, so first-request latency
    excludes one-time kernel compilation / allocator growth."""
    tok = engine.backend.tokenizer
    params = SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True)
    handles = [
        engine.submit(tok.encode(p) or [0], params, 600.0, "_warmup")
        for p in ("Hello, world.", "Warm up the batched decode path, please.")
    ]
    for h in handles:
        async for ev in h.events():
            if isinstance(ev, FinishEvent) and ev.reason == FinishReason.ERROR:
                raise RuntimeError(f"warmup failed: {ev.error}")
    engine.block_manager.reset_prefix_cache()


class AppState:
    def __init__(self, settings: Settings, backend_override: Backend | None) -> None:
        self.settings = settings
        self.status = "starting"  # starting -> loading -> ready | failed -> stopped
        self.error: str | None = None
        self.draining = False
        self.backend: Backend | None = None
        self.engine: Engine | None = None
        self.load_s: float | None = None
        self.metrics = Metrics()
        self.limiter = RateLimiter(settings.rate_limit_rpm, settings.rate_limit_burst)
        self.keys = settings.key_to_tenant()
        self._override = backend_override
        self._load_task: asyncio.Task | None = None

    def start_loading(self) -> None:
        self._load_task = asyncio.create_task(self._load(), name="miniserve-load")

    async def _load(self) -> None:
        self.status = "loading"
        t0 = time.perf_counter()
        try:
            backend = self._override or await asyncio.to_thread(load_backend, self.settings)
            self.backend = backend
            engine = Engine(backend, engine_config(self.settings))
            engine.start()
            self.engine = engine
            if self.settings.warmup:
                await _warmup(engine)
            engine.metrics = self.metrics  # warmup traffic is not reported
            self.load_s = round(time.perf_counter() - t0, 3)
            self.status = "ready"
            log.info(
                "ready",
                extra={
                    "backend": backend.name,
                    "model": backend.model_id,
                    "device": backend.device,
                    "dtype": backend.dtype,
                    "kv_blocks": backend.num_kv_blocks,
                    "block_size": backend.block_size,
                    "load_s": self.load_s,
                },
            )
        except Exception as e:
            self.status, self.error = "failed", f"{type(e).__name__}: {e}"
            log.exception("model load failed")

    def begin_drain(self) -> None:
        """Stop accepting work (readiness fails) while in-flight requests finish."""
        if not self.draining:
            self.draining = True
            log.info("draining", extra={"active": self.engine.num_active if self.engine else 0})

    async def shutdown(self) -> None:
        self.draining = True
        if self._load_task is not None and not self._load_task.done():
            self._load_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._load_task
        if self.engine is not None:
            await self.engine.stop()
        if self.backend is not None and self._override is None:  # injected: caller owns it
            self.backend.close()
        self.status = "stopped"

    def require_ready(self) -> tuple[Engine, Backend]:
        if self.draining:
            raise APIError(
                503, "server is draining", "service_unavailable", headers={"Retry-After": "5"}
            )
        if self.status != "ready" or self.engine is None or self.backend is None:
            msg = f"model not ready (status={self.status})"
            if self.error:
                msg += f": {self.error}"
            raise APIError(503, msg, "service_unavailable", headers={"Retry-After": "5"})
        return self.engine, self.backend

    def engine_alive(self) -> bool:
        return self.engine is None or (
            self.engine._thread is not None and self.engine._thread.is_alive()
        )

    def info(self) -> dict[str, Any]:
        b, e = self.backend, self.engine
        out = {
            "status": "draining" if self.draining and self.status == "ready" else self.status,
            "error": self.error,
            "backend": b.name if b else self.settings.backend,
            "model": b.model_id if b else None,
            "device": b.device if b else None,
            "dtype": b.dtype if b else None,
            "load_s": self.load_s,
            "active_requests": e.num_active if e else 0,
            "version": __version__,
        }
        if e is not None and e.fatal_error:
            out["engine_error"] = e.fatal_error
        return out


class _SSEResponse(StreamingResponse):
    """Streaming response that always cancels its request when the response ends.

    Starlette cancels the streaming task on client disconnect, but if the body generator is parked
    at a ``yield`` it is only closed later by garbage collection; this makes cancellation prompt
    and deterministic regardless of where the generator was suspended.
    """

    def __init__(self, content, handle: RequestHandle) -> None:
        super().__init__(
            content,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Request-ID": handle.id,
            },
        )
        self._handle = handle

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._handle.cancel()


async def _cancel_on_disconnect(request: Request, handle: RequestHandle) -> None:
    """Non-streaming: the body is already read, so the next ASGI message is a disconnect."""
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            if handle.cancel():
                log.info("client disconnected", extra={"request_id": handle.id})
            return


class _Formatter:
    """Builds OpenAI-shaped bodies/chunks for text completions or chat completions."""

    def __init__(self, kind: Kind, handle: RequestHandle, model: str) -> None:
        self.kind, self.handle, self.model = kind, handle, model
        self.created = int(time.time())

    def _base(self, obj: str) -> dict[str, Any]:
        return {"id": self.handle.id, "object": obj, "created": self.created, "model": self.model}

    @staticmethod
    def _extras(out: dict[str, Any], f: FinishEvent) -> dict[str, Any]:
        out["usage"] = {
            "prompt_tokens": f.prompt_tokens,
            "completion_tokens": f.completion_tokens,
            "total_tokens": f.prompt_tokens + f.completion_tokens,
            "prompt_tokens_details": {"cached_tokens": f.cached_tokens},
        }
        out["miniserve"] = {"timings": f.timings, "preemptions": f.preemptions, "error": f.error}
        if f.draft_tokens:
            out["miniserve"]["speculative"] = {
                "draft_tokens": f.draft_tokens,
                "accepted_tokens": f.accepted_tokens,
            }
        return out

    def role_chunk(self) -> dict[str, Any] | None:
        if self.kind != "chat":
            return None
        out = self._base("chat.completion.chunk")
        out["choices"] = [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
                "logprobs": None,
            }
        ]
        return out

    def chunk(self, text: str, f: FinishEvent | None) -> dict[str, Any]:
        reason = str(f.reason) if f else None
        if self.kind == "chat":
            out = self._base("chat.completion.chunk")
            delta = {"content": text} if text else {}
            out["choices"] = [
                {"index": 0, "delta": delta, "finish_reason": reason, "logprobs": None}
            ]
        else:
            out = self._base("text_completion")
            out["choices"] = [{"index": 0, "text": text, "finish_reason": reason, "logprobs": None}]
        return self._extras(out, f) if f else out

    def body(self, text: str, f: FinishEvent) -> dict[str, Any]:
        if self.kind == "chat":
            out = self._base("chat.completion")
            out["choices"] = [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": str(f.reason),
                    "logprobs": None,
                }
            ]
        else:
            out = self._base("text_completion")
            out["choices"] = [
                {"index": 0, "text": text, "finish_reason": str(f.reason), "logprobs": None}
            ]
        return self._extras(out, f)


def _sse(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def create_app(settings: Settings | None = None, backend: Backend | None = None) -> FastAPI:
    """Build the app. ``backend`` injects a pre-built backend (tests); otherwise one is loaded."""
    settings = settings or Settings()
    state = AppState(settings, backend)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state.start_loading()
        try:
            yield
        finally:
            await state.shutdown()

    app = FastAPI(title="MiniServe", version=__version__, lifespan=lifespan)
    app.state.miniserve = state
    # Reject oversized bodies before FastAPI buffers and parses them (memory-exhaustion guard).
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_bytes)

    @app.exception_handler(APIError)
    async def _api_error(request: Request, exc: APIError) -> JSONResponse:
        return exc.response()

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"loc": e.get("loc"), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()
        ]
        err = {
            "message": "invalid request",
            "type": "invalid_request_error",
            "code": "validation_error",
        }
        return JSONResponse({"error": {**err, "details": details}}, 422)

    async def tenant_of(request: Request) -> str:
        """Bearer-token auth (if keys are configured); returns the tenant for this request."""
        if not settings.auth_enabled:
            return request.client.host if request.client else "anonymous"
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() == "bearer":
            for key, tenant in state.keys.items():
                if hmac.compare_digest(token.encode(), key.encode()):
                    return tenant
        state.metrics.rejected.labels("unauthorized").inc()
        raise APIError(401, "invalid or missing API key", "authentication_error")

    # ------------------------------------------------------------------ probes / metrics
    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        """Liveness: process up, model load has not failed, engine thread alive."""
        ok = state.status != "failed" and state.engine_alive()
        return JSONResponse(state.info(), 200 if ok else 503)

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        """Readiness: model loaded + warmed up and not draining; safe to route traffic here."""
        ok = state.status == "ready" and not state.draining
        return JSONResponse(state.info(), 200 if ok else 503)

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(
            state.metrics.render(state.engine),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.get("/v1/models", dependencies=[Depends(tenant_of)])
    async def models() -> dict[str, Any]:
        _, b = state.require_ready()
        return {
            "object": "list",
            "data": [
                {
                    "id": b.model_id,
                    "object": "model",
                    "owned_by": "miniserve",
                    "max_model_len": b.max_context,
                }
            ],
        }

    @app.get("/v1/engine", dependencies=[Depends(tenant_of)])
    async def engine_state() -> dict[str, Any]:
        """Debug view of scheduler / KV-cache state (approximate: read across threads)."""
        e, b = state.require_ready()
        bm = e.block_manager
        return {
            "backend": b.name,
            "model": b.model_id,
            "device": b.device,
            "dtype": b.dtype,
            "config": vars(e.config),
            "steps": e.num_steps,
            "running": e.scheduler.num_running,
            "waiting": e.scheduler.num_waiting,
            "preemptions": e.scheduler.num_preemptions,
            "speculative": e.speculation_stats(),
            "kv_cache": {
                "block_size": bm.block_size,
                "blocks_total": bm.num_blocks,
                "blocks_used": bm.num_used_blocks,
                "blocks_cached": bm.num_cached_blocks,
                "prefix_caching": bm.enable_prefix_caching,
                "prefix_queried_tokens": bm.stats.queried_tokens,
                "prefix_hit_tokens": bm.stats.hit_tokens,
                "evictions": bm.stats.evictions,
            },
            "memory": b.memory_stats(),
        }

    @app.get("/v1/engine/blocks", dependencies=[Depends(tenant_of)])
    async def engine_blocks() -> dict[str, Any]:
        """Per-block KV cache state for the dashboard: '.' free, 'c' cached (evictable),
        '1'-'9' in use with that many sharers (capped at 9)."""
        e, _ = state.require_ready()
        bm = e.block_manager
        for _ in range(10):  # the worker may mutate the dicts mid-copy; retry the snapshot
            try:
                refs = list(bm._ref)
                cached = set(bm._evictable)
                running = list(e.scheduler.running)
                break
            except RuntimeError:
                await asyncio.sleep(0.001)
        else:
            raise APIError(503, "engine busy, retry", "service_unavailable")
        states = "".join(
            str(min(r, 9)) if r else ("c" if b in cached else ".") for b, r in enumerate(refs)
        )
        seqs = [
            {
                "id": s.request_id[-6:],
                "tenant": s.tenant,
                "tokens": s.num_tokens,
                "computed": s.num_computed,
                "blocks": len(s.block_ids),
                "cached_tokens": s.num_cached_tokens,
                "preemptions": s.num_preemptions,
            }
            for s in running
        ]
        return {
            "block_size": bm.block_size,
            "states": states,
            "running": seqs,
            "waiting": e.scheduler.num_waiting,
            "steps": e.num_steps,
            "preemptions": e.scheduler.num_preemptions,
            "prefix_hit_tokens": bm.stats.hit_tokens,
            "prefix_queried_tokens": bm.stats.queried_tokens,
            "speculative": e.speculation_stats(),
        }

    @app.get("/dashboard", include_in_schema=False)
    async def dashboard() -> HTMLResponse:
        return HTMLResponse((Path(__file__).parent / "static" / "dashboard.html").read_text())

    @app.post("/v1/requests/{request_id}/cancel")
    async def cancel_request(request_id: str, tenant: str = Depends(tenant_of)) -> dict[str, Any]:
        # Works while draining (owners may still abandon in-flight work). Another tenant's
        # request is reported as not found rather than forbidden: ids are not disclosed.
        engine = state.engine
        handle = engine.get(request_id) if engine is not None else None
        if handle is None or handle.tenant != tenant:
            raise APIError(404, f"no active request {request_id!r}", code="request_not_found")
        was = handle.state
        handle.cancel()
        return {"id": request_id, "cancelled": True, "state": was}

    # ------------------------------------------------------------------ generation
    def reject(
        status: int,
        reason: str,
        message: str,
        type_: str = "invalid_request_error",
        code: str | None = None,
        headers=None,
    ) -> APIError:
        state.metrics.rejected.labels(reason).inc()
        return APIError(status, message, type_, code, headers)

    async def generate(request: Request, body, tenant: str, kind: Kind):
        retry_after = state.limiter.acquire(tenant)
        if retry_after > 0:
            raise reject(
                429,
                "rate_limited",
                "rate limit exceeded",
                "rate_limit_error",
                headers={"Retry-After": str(math.ceil(retry_after))},
            )
        engine, backend = state.require_ready()
        if body.model is not None and body.model != backend.model_id:
            raise APIError(404, f"model {body.model!r} is not served here", code="model_not_found")

        if kind == "chat":
            chars = sum(len(m.content) for m in body.messages)
        else:
            chars = len(body.prompt)
        if chars > settings.max_prompt_chars:
            raise reject(400, "too_long", f"prompt exceeds {settings.max_prompt_chars} characters")
        cap = settings.max_new_tokens_cap
        requested = getattr(body, "max_completion_tokens", None) or body.max_tokens
        max_tokens = requested or min(settings.default_max_tokens, cap)
        if max_tokens > cap:
            raise reject(
                400,
                "too_long",
                f"max_tokens={max_tokens} exceeds server cap {cap}",
                code="max_tokens",
            )
        if kind == "chat":
            try:
                prompt_ids = backend.tokenizer.apply_chat_template(
                    [m.model_dump() for m in body.messages]
                )
            except ValueError as e:
                raise APIError(400, str(e), code="chat_template") from e
        else:
            prompt_ids = backend.tokenizer.encode(body.prompt)
        if not prompt_ids:
            raise APIError(400, "prompt is empty after tokenization")
        if len(prompt_ids) > settings.max_prompt_tokens:
            raise reject(
                400,
                "too_long",
                f"prompt has {len(prompt_ids)} tokens; limit is {settings.max_prompt_tokens}",
                code="prompt_too_long",
            )
        if len(prompt_ids) + max_tokens > backend.max_context:
            raise reject(
                400,
                "too_long",
                f"prompt ({len(prompt_ids)}) + max_tokens ({max_tokens}) "
                f"exceeds the model context window ({backend.max_context})",
                code="context_length_exceeded",
            )
        if (why := engine.check_request_fits(len(prompt_ids), max_tokens)) is not None:
            raise reject(400, "too_long", why, code="exceeds_kv_capacity")

        params = SamplingParams(
            max_tokens, body.temperature, body.top_p, body.seed, body.ignore_eos, body.stop_tuple
        )
        timeout_s = min(body.timeout_s or math.inf, settings.request_timeout_s)
        try:
            handle = engine.submit(prompt_ids, params, timeout_s, tenant)
        except EngineOverloaded as e:
            if e.per_tenant:
                raise reject(
                    429,
                    "tenant_queue_full",
                    str(e),
                    "rate_limit_error",
                    headers={"Retry-After": "1"},
                ) from e
            raise reject(
                503,
                "queue_full",
                f"server busy: {e}",
                "overloaded_error",
                headers={"Retry-After": "1"},
            ) from e
        except EngineClosed as e:
            raise APIError(503, str(e), "service_unavailable") from e

        fmt = _Formatter(kind, handle, backend.model_id)
        if body.stream:

            async def sse():
                try:
                    if (first := fmt.role_chunk()) is not None:
                        yield _sse(first)
                    async for ev in handle.events():
                        if isinstance(ev, TokenEvent):
                            if ev.text:
                                yield _sse(fmt.chunk(ev.text, None))
                        else:
                            final = fmt.chunk(ev.text, ev)
                            if ev.error:
                                final["error"] = {"message": ev.error, "type": "server_error"}
                            yield _sse(final)
                    yield "data: [DONE]\n\n"
                finally:
                    handle.cancel()

            return _SSEResponse(sse(), handle)

        watcher = asyncio.create_task(_cancel_on_disconnect(request, handle))
        parts: list[str] = []
        finish: FinishEvent | None = None
        try:
            async for ev in handle.events():
                parts.append(ev.text)
                if isinstance(ev, FinishEvent):
                    finish = ev
        finally:
            watcher.cancel()
            handle.cancel()
        assert finish is not None
        result = fmt.body("".join(parts), finish)
        if finish.reason == FinishReason.TIMEOUT:
            raise APIError(504, "request timed out", "timeout_error", extra={"partial": result})
        if finish.reason == FinishReason.ERROR:
            raise APIError(
                500, f"generation failed: {finish.error}", "server_error", extra={"partial": result}
            )
        return JSONResponse(result, headers={"X-Request-ID": handle.id})

    @app.post("/v1/completions")
    async def completions(
        body: CompletionRequest, request: Request, tenant: str = Depends(tenant_of)
    ):
        return await generate(request, body, tenant, "text")

    @app.post("/v1/chat/completions")
    async def chat_completions(
        body: ChatCompletionRequest, request: Request, tenant: str = Depends(tenant_of)
    ):
        return await generate(request, body, tenant, "chat")

    return app
