"""FastAPI app: model lifecycle, auth + limits, validation, completions (JSON and SSE)."""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import itertools
import json
import logging
import math
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from miniserve import __version__
from miniserve.backends import Backend, SamplingParams, load_backend
from miniserve.config import Settings
from miniserve.engine import (
    Engine,
    EngineClosed,
    EngineOverloaded,
    FinishEvent,
    FinishReason,
    RequestHandle,
    TokenEvent,
)
from miniserve.ratelimit import RateLimiter
from miniserve.schemas import CompletionRequest

log = logging.getLogger("miniserve")


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


def _warmup(backend: Backend) -> None:
    """Run a few decode steps so first-request latency excludes one-time kernel/alloc setup."""
    ids = backend.tokenizer.encode("Hello, world.") or [0]
    it = backend.generate(ids, SamplingParams(max_tokens=4, temperature=0.0))
    try:
        for _ in itertools.islice(it, 4):
            pass
    finally:
        it.close()


class AppState:
    def __init__(self, settings: Settings, backend_override: Backend | None) -> None:
        self.settings = settings
        self.status = "starting"  # starting -> loading -> ready | failed -> stopped
        self.error: str | None = None
        self.backend: Backend | None = None
        self.engine: Engine | None = None
        self.load_s: float | None = None
        self.limiter = RateLimiter(settings.rate_limit_rpm, settings.rate_limit_burst)
        self._override = backend_override
        self._load_task: asyncio.Task | None = None

    def start_loading(self) -> None:
        self._load_task = asyncio.create_task(self._load(), name="miniserve-load")

    async def _load(self) -> None:
        self.status = "loading"
        t0 = time.perf_counter()
        try:
            backend = self._override or await asyncio.to_thread(load_backend, self.settings)
            if self.settings.warmup:
                await asyncio.to_thread(_warmup, backend)
            engine = Engine(backend, self.settings.max_pending)
            engine.start()
            self.backend, self.engine = backend, engine
            self.load_s = round(time.perf_counter() - t0, 3)
            self.status = "ready"
            log.info(
                "ready backend=%s model=%s device=%s load+warmup=%.2fs",
                backend.name,
                backend.model_id,
                backend.device,
                self.load_s,
            )
        except Exception as e:
            self.status, self.error = "failed", f"{type(e).__name__}: {e}"
            log.exception("model load failed")

    async def shutdown(self) -> None:
        if self._load_task is not None and not self._load_task.done():
            self._load_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._load_task
        if self.engine is not None:
            await self.engine.stop()
        if self.backend is not None and self._override is None:  # injected backends: caller owns
            self.backend.close()
        self.status = "stopped"

    def require_ready(self) -> tuple[Engine, Backend]:
        if self.status != "ready" or self.engine is None or self.backend is None:
            msg = f"model not ready (status={self.status})"
            if self.error:
                msg += f": {self.error}"
            raise APIError(503, msg, "service_unavailable", headers={"Retry-After": "5"})
        return self.engine, self.backend

    def info(self) -> dict[str, Any]:
        b = self.backend
        return {
            "status": self.status,
            "error": self.error,
            "backend": b.name if b else self.settings.backend,
            "model": b.model_id if b else None,
            "device": b.device if b else None,
            "dtype": getattr(b, "dtype", None) if b else None,
            "load_s": self.load_s,
            "active_requests": self.engine.num_active if self.engine else 0,
            "version": __version__,
        }


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
                log.info("client disconnected, cancelled id=%s", handle.id)
            return


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

    async def client_identity(request: Request) -> str:
        """Bearer-token auth (if a key is configured); returns the rate-limit key."""
        if settings.api_key:
            scheme, _, token = request.headers.get("authorization", "").partition(" ")
            if scheme.lower() != "bearer" or not hmac.compare_digest(
                token.encode(), settings.api_key.encode()
            ):
                raise APIError(401, "invalid or missing API key", "authentication_error")
        return request.client.host if request.client else "unknown"

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        """Liveness: the process is up and model loading has not failed."""
        return JSONResponse(state.info(), 503 if state.status == "failed" else 200)

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        """Readiness: model loaded and warmed up; safe to route traffic here."""
        return JSONResponse(state.info(), 200 if state.status == "ready" else 503)

    @app.get("/v1/models", dependencies=[Depends(client_identity)])
    async def models() -> dict[str, Any]:
        _, b = state.require_ready()
        return {
            "object": "list",
            "data": [{"id": b.model_id, "object": "model", "owned_by": "miniserve"}],
        }

    @app.post("/v1/requests/{request_id}/cancel", dependencies=[Depends(client_identity)])
    async def cancel_request(request_id: str) -> dict[str, Any]:
        engine, _ = state.require_ready()
        handle = engine.get(request_id)
        if handle is None:
            raise APIError(404, f"no active request {request_id!r}", code="request_not_found")
        was = handle.state
        handle.cancel()
        return {"id": request_id, "cancelled": True, "state": was}

    @app.post("/v1/completions")
    async def completions(
        body: CompletionRequest, request: Request, client: str = Depends(client_identity)
    ):
        retry_after = state.limiter.acquire(client)
        if retry_after > 0:
            raise APIError(
                429,
                "rate limit exceeded",
                "rate_limit_error",
                headers={"Retry-After": str(math.ceil(retry_after))},
            )
        engine, backend = state.require_ready()

        if body.model is not None and body.model != backend.model_id:
            raise APIError(404, f"model {body.model!r} is not served here", code="model_not_found")
        if len(body.prompt) > settings.max_prompt_chars:
            raise APIError(400, f"prompt exceeds {settings.max_prompt_chars} characters")
        cap = settings.max_new_tokens_cap
        max_tokens = body.max_tokens or min(settings.default_max_tokens, cap)
        if max_tokens > cap:
            raise APIError(
                400, f"max_tokens={max_tokens} exceeds server cap {cap}", code="max_tokens"
            )

        prompt_ids = backend.tokenizer.encode(body.prompt)
        if not prompt_ids:
            raise APIError(400, "prompt is empty after tokenization")
        if len(prompt_ids) > settings.max_prompt_tokens:
            raise APIError(
                400,
                f"prompt has {len(prompt_ids)} tokens; limit is {settings.max_prompt_tokens}",
                code="prompt_too_long",
            )
        if len(prompt_ids) + max_tokens > backend.max_context:
            raise APIError(
                400,
                f"prompt ({len(prompt_ids)}) + max_tokens ({max_tokens}) exceeds the model context "
                f"window ({backend.max_context})",
                code="context_length_exceeded",
            )

        params = SamplingParams(max_tokens, body.temperature, body.top_p, body.seed)
        timeout_s = min(body.timeout_s or math.inf, settings.request_timeout_s)
        try:
            handle = engine.submit(prompt_ids, params, timeout_s)
        except EngineOverloaded as e:
            raise APIError(
                503, f"server busy: {e}", "overloaded_error", headers={"Retry-After": "1"}
            ) from e
        except EngineClosed as e:
            raise APIError(503, str(e), "service_unavailable") from e

        created = int(time.time())

        def chunk(text: str, finish: FinishEvent | None) -> dict[str, Any]:
            out: dict[str, Any] = {
                "id": handle.id,
                "object": "text_completion",
                "created": created,
                "model": backend.model_id,
                "choices": [
                    {
                        "index": 0,
                        "text": text,
                        "finish_reason": str(finish.reason) if finish else None,
                        "logprobs": None,
                    }
                ],
            }
            if finish is not None:
                out["usage"] = {
                    "prompt_tokens": finish.prompt_tokens,
                    "completion_tokens": finish.completion_tokens,
                    "total_tokens": finish.prompt_tokens + finish.completion_tokens,
                }
                out["miniserve"] = {"timings": finish.timings, "error": finish.error}
            return out

        if body.stream:

            async def sse():
                try:
                    async for ev in handle.events():
                        if isinstance(ev, TokenEvent):
                            if ev.text:
                                yield f"data: {json.dumps(chunk(ev.text, None))}\n\n"
                        else:
                            yield f"data: {json.dumps(chunk(ev.text, ev))}\n\n"
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
        result = chunk("".join(parts), finish)
        if finish.reason == FinishReason.TIMEOUT:
            raise APIError(504, "request timed out", "timeout_error", extra={"partial": result})
        if finish.reason == FinishReason.ERROR:
            raise APIError(
                500, f"generation failed: {finish.error}", "server_error", extra={"partial": result}
            )
        return JSONResponse(result, headers={"X-Request-ID": handle.id})

    return app
