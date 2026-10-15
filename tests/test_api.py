"""HTTP API tests (in-process TestClient, toy backend)."""

import time

import pytest

from miniserve.backends.toy import ToyBackend

from conftest import app_client, make_settings, parse_sse


def complete(client, **body):
    return client.post("/v1/completions", json=body)


def test_non_stream_completion_shape(toy_client):
    r = complete(toy_client, prompt="hello world", max_tokens=50)
    assert r.status_code == 200
    body = r.json()
    assert r.headers["x-request-id"] == body["id"]
    assert body["object"] == "text_completion" and body["model"] == "toy-echo"
    assert body["choices"][0] == {
        "index": 0,
        "text": "hello world",
        "finish_reason": "stop",
        "logprobs": None,
    }
    assert body["usage"] == {"prompt_tokens": 11, "completion_tokens": 11, "total_tokens": 22}
    assert set(body["miniserve"]["timings"]) == {
        "queue_ms",
        "ttft_ms",
        "total_ms",
        "decode_tokens_per_s",
    }


def test_max_tokens_is_enforced_exactly(toy_client):
    body = complete(toy_client, prompt="abcdefghij", max_tokens=4).json()
    assert body["choices"][0]["text"] == "abcd"
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 4


def test_default_max_tokens_applies():
    with app_client(ToyBackend(), default_max_tokens=3) as c:
        body = complete(c, prompt="abcdefgh").json()
        assert body["usage"]["completion_tokens"] == 3


@pytest.mark.parametrize("prompt", ["hello world", "héllo wörld 🙂 世界 👩‍💻", "a\nb\tc"])
def test_stream_matches_non_stream(toy_client, prompt):
    full = complete(toy_client, prompt=prompt, max_tokens=200).json()
    r = complete(toy_client, prompt=prompt, max_tokens=200, stream=True)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    chunks, done = parse_sse(r.text)
    assert done
    assert all(c["id"] == r.headers["x-request-id"] for c in chunks)
    assert all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1])
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"] == full["usage"]
    assert "".join(c["choices"][0]["text"] for c in chunks) == full["choices"][0]["text"]
    assert all("�" not in c["choices"][0]["text"] for c in chunks)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"prompt": ""},
        {"prompt": "hi", "temperature": 2.5},
        {"prompt": "hi", "temperature": -0.1},
        {"prompt": "hi", "top_p": 0},
        {"prompt": "hi", "max_tokens": 0},
        {"prompt": "hi", "seed": -1},
        {"prompt": "hi", "timeout_s": 0},
        {"prompt": "hi", "n": 2},  # unsupported OpenAI field -> rejected, not silently ignored
        {"prompt": ["batched", "prompts"]},
    ],
)
def test_validation_errors(toy_client, body):
    r = complete(toy_client, **body)
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "validation_error"


def test_limits_return_400():
    with app_client(
        ToyBackend(max_context=40), max_new_tokens_cap=16, max_prompt_tokens=30, max_prompt_chars=50
    ) as c:
        r = complete(c, prompt="hi", max_tokens=17)
        assert r.status_code == 400 and r.json()["error"]["code"] == "max_tokens"
        r = complete(c, prompt="x" * 51)
        assert r.status_code == 400 and "characters" in r.json()["error"]["message"]
        r = complete(c, prompt="é" * 20)  # 20 chars but 40 byte tokens
        assert r.status_code == 400 and r.json()["error"]["code"] == "prompt_too_long"
        r = complete(c, prompt="x" * 30, max_tokens=16)  # 30 + 16 > 40
        assert r.status_code == 400 and r.json()["error"]["code"] == "context_length_exceeded"
        assert complete(c, prompt="x" * 24, max_tokens=16).status_code == 200  # exactly 40


def test_wrong_model_is_404(toy_client):
    assert complete(toy_client, prompt="hi", model="gpt-4").status_code == 404
    assert complete(toy_client, prompt="hi", model="toy-echo").status_code == 200
    assert toy_client.get("/v1/models").json()["data"][0]["id"] == "toy-echo"


def test_auth_required_when_key_set():
    with app_client(ToyBackend(), api_key="s3cret") as c:
        assert c.get("/healthz").status_code == 200  # probes stay unauthenticated
        assert complete(c, prompt="hi").status_code == 401
        r = c.post("/v1/completions", json={"prompt": "hi"}, headers={"Authorization": "Bearer no"})
        assert r.status_code == 401 and r.json()["error"]["type"] == "authentication_error"
        ok = {"Authorization": "Bearer s3cret"}
        assert c.post("/v1/completions", json={"prompt": "hi"}, headers=ok).status_code == 200
        assert c.get("/v1/models").status_code == 401
        assert c.post("/v1/requests/x/cancel").status_code == 401


def test_rate_limit_returns_429_with_retry_after():
    with app_client(ToyBackend(), rate_limit_rpm=60, rate_limit_burst=2) as c:
        assert complete(c, prompt="a").status_code == 200
        assert complete(c, prompt="b").status_code == 200
        r = complete(c, prompt="c")
        assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1


def test_not_ready_while_loading_then_ready():
    # warmup runs the slow toy prefill (13 bytes x 40 ms ~ 0.5 s) before the engine starts
    from fastapi.testclient import TestClient

    from miniserve.server import create_app

    app = create_app(make_settings(warmup=True), ToyBackend(prefill_ms_per_token=40))
    with TestClient(app) as c:
        r = c.get("/readyz")
        assert r.status_code == 503 and r.json()["status"] == "loading"
        assert c.get("/healthz").status_code == 200
        r = complete(c, prompt="hi")
        assert r.status_code == 503 and r.headers["retry-after"] == "5"
        deadline = time.monotonic() + 5
        while c.get("/readyz").status_code != 200:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert c.get("/readyz").json()["load_s"] >= 0.4
        assert complete(c, prompt="hi").status_code == 200


def test_failed_load_is_reported():
    from fastapi.testclient import TestClient

    from miniserve.server import create_app

    class Broken(ToyBackend):
        def generate(self, prompt_ids, params):
            raise RuntimeError("weights corrupted")
            yield  # pragma: no cover

    with TestClient(create_app(make_settings(warmup=True), Broken())) as c:
        deadline = time.monotonic() + 5
        while c.get("/readyz").json()["status"] == "loading":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert c.get("/healthz").status_code == 503
        assert "weights corrupted" in c.get("/readyz").json()["error"]
        assert complete(c, prompt="hi").status_code == 503


def test_timeout_non_stream_is_504_with_partial():
    with app_client(ToyBackend(decode_ms_per_token=20)) as c:
        r = complete(c, prompt="z" * 100, timeout_s=0.2)
        assert r.status_code == 504
        partial = r.json()["partial"]
        assert partial["choices"][0]["finish_reason"] == "timeout"
        assert 0 < partial["usage"]["completion_tokens"] < 100


def test_timeout_stream_is_reported_in_band():
    with app_client(ToyBackend(decode_ms_per_token=20)) as c:
        r = complete(c, prompt="z" * 100, timeout_s=0.2, stream=True)
        chunks, done = parse_sse(r.text)
        assert r.status_code == 200 and done
        assert chunks[-1]["choices"][0]["finish_reason"] == "timeout"


def test_server_timeout_caps_client_timeout():
    with app_client(ToyBackend(decode_ms_per_token=20), request_timeout_s=0.2) as c:
        t0 = time.monotonic()
        r = complete(c, prompt="z" * 100, timeout_s=999)
        assert r.status_code == 504 and time.monotonic() - t0 < 1.0


def test_cancel_unknown_request_is_404(toy_client):
    r = toy_client.post("/v1/requests/cmpl-nope/cancel")
    assert r.status_code == 404 and r.json()["error"]["code"] == "request_not_found"


def test_bind_safety():
    with pytest.raises(RuntimeError, match="Refusing to bind"):
        make_settings(host="0.0.0.0").check_bind_safety()
    make_settings(host="0.0.0.0", api_key="k").check_bind_safety()
    make_settings(host="0.0.0.0", allow_unauthenticated_public=True).check_bind_safety()
    for host in ("127.0.0.1", "localhost", "::1"):
        make_settings(host=host).check_bind_safety()


def test_injected_backend_survives_app_shutdown():
    class Tracking(ToyBackend):
        closed = False

        def close(self):
            self.closed = True

    backend = Tracking()
    for _ in range(2):  # the same backend instance can serve two app lifecycles
        with app_client(backend) as c:
            assert complete(c, prompt="hi").status_code == 200
    assert not backend.closed
