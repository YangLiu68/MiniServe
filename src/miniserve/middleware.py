"""ASGI middleware: request body size limit.

Uvicorn reads request bodies of any size; FastAPI then buffers them to parse JSON *before* any
of our validation runs, so without a cap a single multi-GB body can exhaust memory long before
``max_prompt_chars`` is checked. This rejects oversized requests with 413: up front when
``Content-Length`` says so (the HTTP parser guarantees the body matches it), and for chunked
uploads without a length by buffering at most ``max_bytes`` *before* the app runs, then replaying
the buffered body. (Raising from inside ``receive`` instead does not work: FastAPI's body parsing
catches any exception there and answers 400 "error parsing the body".)
"""

from __future__ import annotations

import json


class BodySizeLimitMiddleware:
    def __init__(self, app, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def _reject(self, send) -> None:
        body = json.dumps(
            {
                "error": {
                    "message": f"request body exceeds {self.max_bytes} bytes",
                    "type": "invalid_request_error",
                    "code": "request_too_large",
                }
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        for name, value in scope.get("headers", ()):
            if name == b"content-length":
                try:
                    too_big = int(value) > self.max_bytes
                except ValueError:
                    too_big = False
                if too_big:
                    return await self._reject(send)
                return await self.app(scope, receive, send)
        # No Content-Length (chunked or bodiless): read up to the limit before the app sees it.
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] != "http.request":  # client disconnected mid-upload
                return
            body = message.get("body", b"")
            size += len(body)
            if size > self.max_bytes:
                return await self._reject(send)
            chunks.append(body)
            if not message.get("more_body", False):
                break
        buffered = {"type": "http.request", "body": b"".join(chunks), "more_body": False}

        async def replay():
            nonlocal buffered
            if buffered is not None:
                message, buffered = buffered, None
                return message
            return await receive()  # afterwards: disconnect notifications

        await self.app(scope, replay, send)
