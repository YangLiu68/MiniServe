from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from miniserve.middleware import BodySizeLimitMiddleware


def make_client(limit=100):
    app = FastAPI()

    @app.post("/echo")
    async def echo(request: Request):
        return {"n": len(await request.body())}

    app.add_middleware(BodySizeLimitMiddleware, max_bytes=limit)
    return TestClient(app)


def test_small_bodies_pass():
    assert make_client().post("/echo", content=b"x" * 100).json() == {"n": 100}


def test_declared_oversize_is_rejected_before_reading():
    r = make_client().post("/echo", content=b"x" * 101)
    assert r.status_code == 413 and r.json()["error"]["code"] == "request_too_large"


def test_chunked_oversize_without_content_length_is_rejected():
    def chunks():
        for _ in range(10):
            yield b"y" * 30  # 300 bytes, no Content-Length header

    r = make_client().post("/echo", content=chunks())
    assert r.status_code == 413


def test_chunked_oversize_to_the_real_api_is_413_with_openai_error():
    """Regression: raising inside receive() was swallowed by FastAPI's body parsing (400)."""
    from miniserve.backends.toy import ToyBackend

    from conftest import app_client

    def chunks():
        for _ in range(100):
            yield b"x" * 2000  # 200 kB, no Content-Length

    with app_client(ToyBackend(), max_request_bytes=100_000) as c:
        r = c.post(
            "/v1/completions", content=chunks(), headers={"content-type": "application/json"}
        )
        assert r.status_code == 413 and r.json()["error"]["code"] == "request_too_large"
        ok = c.post(
            "/v1/completions",
            content=iter([b'{"prompt": "chunked', b' but small", "max_tokens": 3}']),
            headers={"content-type": "application/json"},
        )
        assert ok.status_code == 200 and ok.json()["choices"][0]["text"] == "chu"


def test_get_requests_unaffected():
    c = make_client()
    assert c.get("/echo").status_code == 405  # route exists, method not allowed: passthrough
