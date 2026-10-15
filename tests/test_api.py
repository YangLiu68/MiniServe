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
    usage = body["usage"]
    assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (
        11,
        11,
        22,
    )
    assert usage["prompt_tokens_details"] == {"cached_tokens": 0}
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
    su, fu = chunks[-1]["usage"], full["usage"]
    assert (su["prompt_tokens"], su["completion_tokens"]) == (
        fu["prompt_tokens"],
        fu["completion_tokens"],
    )
    # the second identical prompt is served from the prefix cache (full 16-token blocks)
    assert su["prompt_tokens_details"]["cached_tokens"] == (fu["prompt_tokens"] - 1) // 16 * 16
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
        {"prompt": "hi", "stop": ["a", "b", "c", "d", "e"]},
        {"prompt": "hi", "stop": ""},
        {"prompt": "hi", "stop": ["x" * 65]},
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
    # warmup runs two short requests through the engine with a slow toy prefill (~2 s)
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
        def execute(self, batch):
            raise RuntimeError("weights corrupted")

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


def test_chat_completion_non_stream_and_stream(toy_client):
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi 🙂"}]
    r = toy_client.post("/v1/chat/completions", json={"messages": msgs, "max_tokens": 200})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion"
    msg = body["choices"][0]["message"]
    # toy template + echo: the "model" repeats the rendered conversation
    assert msg == {"role": "assistant", "content": "system: be brief\nuser: hi 🙂\nassistant: "}
    r = toy_client.post(
        "/v1/chat/completions", json={"messages": msgs, "max_tokens": 200, "stream": True}
    )
    chunks, done = parse_sse(r.text)
    assert done and chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    streamed = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert streamed == msg["content"]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_chat_validation(toy_client):
    bad = [
        {"messages": []},
        {"messages": [{"role": "tool", "content": "x"}]},
        {"messages": [{"role": "user"}]},
    ]
    for body in bad:
        assert toy_client.post("/v1/chat/completions", json=body).status_code == 422


def test_max_completion_tokens_alias(toy_client):
    r = toy_client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "abcdef"}], "max_completion_tokens": 3},
    )
    assert r.json()["usage"]["completion_tokens"] == 3


def test_stop_and_ignore_eos_over_http(toy_client):
    body = complete(toy_client, prompt="alpha beta gamma", stop=[" gam"], max_tokens=100).json()
    assert body["choices"][0]["text"] == "alpha beta"
    assert body["choices"][0]["finish_reason"] == "stop"
    body = complete(toy_client, prompt="ab", max_tokens=6, ignore_eos=True).json()
    assert body["usage"]["completion_tokens"] == 6
    assert body["choices"][0]["finish_reason"] == "length"


def test_multi_tenant_keys_and_per_tenant_rate_limit():
    with app_client(
        ToyBackend(), api_keys="alice:ka,bob:kb", rate_limit_rpm=60, rate_limit_burst=1
    ) as c:
        alice = {"Authorization": "Bearer ka"}
        bob = {"Authorization": "Bearer kb"}
        assert c.post("/v1/completions", json={"prompt": "a"}, headers=alice).status_code == 200
        assert c.post("/v1/completions", json={"prompt": "a"}, headers=alice).status_code == 429
        # bob has a separate bucket
        assert c.post("/v1/completions", json={"prompt": "b"}, headers=bob).status_code == 200
        assert (
            c.post(
                "/v1/completions", json={"prompt": "b"}, headers={"Authorization": "Bearer kc"}
            ).status_code
            == 401
        )


def test_openai_client_variants_n1_developer_role_and_text_parts(toy_client):
    assert complete(toy_client, prompt="ab", n=1, max_tokens=2).status_code == 200
    r = complete(toy_client, prompt="ab", n=2)
    assert r.status_code == 422  # one choice per request; rejected, not silently ignored
    r = toy_client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "developer", "content": "be brief"},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hi "}, {"type": "text", "text": "there"}],
                },
            ],
            "max_tokens": 200,
        },
    )
    assert r.status_code == 200
    # toy chat template echoes "role: content" lines; developer is mapped to system
    assert (
        r.json()["choices"][0]["message"]["content"]
        == "system: be brief\nuser: hi there\nassistant: "
    )
    image = {"type": "image_url", "image_url": {"url": "http://x/y.png"}}
    r = toy_client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": [image]}]}
    )
    assert r.status_code == 422


def test_prefix_cache_scope_tenant_isolates_tenants():
    system = "a long shared system prompt for everyone. " * 2  # several full blocks
    for scope, bob_hit in (("global", True), ("tenant", False)):
        with app_client(ToyBackend(), api_keys="alice:ka,bob:kb", prefix_cache_scope=scope) as c:

            def cached(key, q):
                r = c.post(
                    "/v1/completions",
                    json={"prompt": system + q, "max_tokens": 1},
                    headers={"Authorization": f"Bearer {key}"},
                )
                return r.json()["usage"]["prompt_tokens_details"]["cached_tokens"]

            assert cached("ka", "1") == 0
            assert (cached("kb", "2") > 0) == bob_hit
            assert cached("ka", "3") > 0  # always shared within a tenant


def test_metrics_endpoint_reports_requests_and_engine_state(toy_client):
    complete(toy_client, prompt="metrics please", max_tokens=5)
    complete(toy_client, prompt="x", max_tokens=99999)  # rejected: over the cap
    text = toy_client.get("/metrics").text
    assert 'miniserve_requests_total{finish_reason="length"} 1.0' in text
    assert "miniserve_time_to_first_token_seconds_count 1.0" in text
    assert 'miniserve_requests_rejected_total{reason="too_long"} 1.0' in text
    assert "miniserve_kv_cache_usage_ratio" in text
    assert "miniserve_generation_tokens_total 5.0" in text
    assert "miniserve_process_resident_memory_bytes" in text


def test_engine_debug_endpoint(toy_client):
    complete(toy_client, prompt="debug", max_tokens=2)
    info = toy_client.get("/v1/engine").json()
    assert info["kv_cache"]["blocks_total"] == 8192
    assert info["steps"] >= 2 and info["running"] == 0
    assert info["config"]["policy"] == "fcfs"


def test_draining_fails_readiness_and_refuses_new_work(toy_client):
    toy_client.app.state.miniserve.begin_drain()
    r = toy_client.get("/readyz")
    assert r.status_code == 503 and r.json()["status"] == "draining"
    assert toy_client.get("/healthz").status_code == 200  # still alive
    assert complete(toy_client, prompt="hi").status_code == 503


def test_request_larger_than_kv_cache_is_400():
    with app_client(ToyBackend(num_kv_blocks=2, block_size=8)) as c:
        r = complete(c, prompt="x" * 10, max_tokens=7)
        assert r.status_code == 400 and r.json()["error"]["code"] == "exceeds_kv_capacity"
        assert complete(c, prompt="x" * 10, max_tokens=6).status_code == 200


def test_block_map_and_dashboard(toy_client):
    system = "shared system prompt! " * 3  # 66 bytes -> 4 full blocks of 16
    for q in ("a?", "b?"):
        complete(toy_client, prompt=system + q, max_tokens=1)
    b = toy_client.get("/v1/engine/blocks").json()
    assert len(b["states"]) == 8192 and set(b["states"]) <= set(".c123456789")
    assert b["states"].count("c") >= 4  # the shared prefix blocks stay cached after finishing
    assert b["prefix_hit_tokens"] == 64 and b["running"] == []
    page = toy_client.get("/dashboard")
    assert page.status_code == 200 and "Paged KV cache" in page.text


def test_block_map_requires_auth_when_enabled():
    with app_client(ToyBackend(), api_key="k") as c:
        assert c.get("/v1/engine/blocks").status_code == 401
        assert c.get("/dashboard").status_code == 200  # static page; it asks for the key


def test_oversized_request_body_is_413():
    with app_client(ToyBackend(), max_request_bytes=4096, max_prompt_chars=10**6) as c:
        r = complete(c, prompt="x" * 5000)
        assert r.status_code == 413 and r.json()["error"]["code"] == "request_too_large"
        assert complete(c, prompt="x" * 3000, max_tokens=1).status_code == 200


def test_env_example_is_a_valid_env_file():
    # Regression: empty values (e.g. MINISERVE_TORCH_THREADS=) crashed startup when the
    # documented `cp .env.example .env` was followed.
    from pathlib import Path

    from miniserve.config import Settings

    example = Path(__file__).parent.parent / ".env.example"
    s = Settings(_env_file=str(example))
    assert s.backend == "native" and s.torch_threads is None and not s.auth_enabled
    documented = {
        ln.split("=")[0].removeprefix("MINISERVE_").lower()
        for ln in example.read_text().splitlines()
        if ln.startswith("MINISERVE_")
    }
    assert documented == set(Settings.model_fields)  # every setting documented, none stale
