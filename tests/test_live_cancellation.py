"""Cancellation against a real uvicorn server over real sockets.

The in-process TestClient cannot simulate a client hanging up, so these tests close actual TCP
connections and then check how many tokens the backend went on to produce.
"""

import threading
import time

import httpx

from miniserve.backends.toy import ToyBackend

from conftest import RecordingBackend, live_server

SLOW = dict(decode_ms_per_token=10)  # ~100 tokens/s


def wait_until(pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_stream_disconnect_stops_generation():
    backend = RecordingBackend(ToyBackend(**SLOW))
    with live_server(backend) as (url, app):
        engine = app.state.miniserve.engine
        with httpx.Client(base_url=url, timeout=10) as client:
            body = {"prompt": "x" * 300, "max_tokens": 300, "stream": True}
            with client.stream("POST", "/v1/completions", json=body) as r:
                assert r.status_code == 200
                received = 0
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        received += 1
                    if received >= 5:
                        break  # leaving the block closes the connection mid-stream
        assert wait_until(lambda: engine.num_active == 0), "request still active after disconnect"
        produced = backend.produced
        time.sleep(0.2)  # would be ~20 more tokens if generation continued
        assert backend.produced == produced
        assert produced < 60, produced  # far fewer than the 300 requested


def test_non_stream_disconnect_stops_generation():
    backend = RecordingBackend(ToyBackend(**SLOW))
    with live_server(backend) as (url, app):
        engine = app.state.miniserve.engine
        try:
            httpx.post(
                f"{url}/v1/completions",
                json={"prompt": "x" * 300, "max_tokens": 300},
                timeout=httpx.Timeout(10, read=0.2),  # client gives up after 0.2 s
            )
        except httpx.ReadTimeout:
            pass
        assert wait_until(lambda: engine.num_active == 0), "request still active after disconnect"
        produced = backend.produced
        time.sleep(0.2)
        assert backend.produced == produced
        assert produced < 80, produced


def test_cancel_endpoint_ends_stream_with_cancelled():
    backend = RecordingBackend(ToyBackend(**SLOW))
    with live_server(backend) as (url, _):
        with httpx.Client(base_url=url, timeout=10) as client:
            body = {"prompt": "x" * 300, "max_tokens": 300, "stream": True}
            with client.stream("POST", "/v1/completions", json=body) as r:
                rid = r.headers["x-request-id"]
                cancel_resp = {}

                def cancel_soon():
                    time.sleep(0.1)
                    cancel_resp["r"] = httpx.post(f"{url}/v1/requests/{rid}/cancel", timeout=5)

                t = threading.Thread(target=cancel_soon)
                t.start()
                lines = [ln for ln in r.iter_lines() if ln.startswith("data: ")]
                t.join()
        assert cancel_resp["r"].status_code == 200
        assert cancel_resp["r"].json()["state"] == "running"
        assert lines[-1] == "data: [DONE]"
        assert '"finish_reason": "cancelled"' in lines[-2]
        assert backend.produced < 60


def test_cancel_is_scoped_to_the_owner_and_works_while_draining():
    """Regressions: any tenant could cancel another tenant's request; cancel returned 503
    while the server was draining."""
    backend = ToyBackend(**SLOW)
    with live_server(backend, api_keys="alice:ka,bob:kb") as (url, app):
        alice, bob = {"Authorization": "Bearer ka"}, {"Authorization": "Bearer kb"}
        with httpx.Client(base_url=url, timeout=10) as client:
            body = {"prompt": "x" * 300, "max_tokens": 300, "stream": True}
            with client.stream("POST", "/v1/completions", json=body, headers=alice) as r:
                rid = r.headers["x-request-id"]
                cancel = f"{url}/v1/requests/{rid}/cancel"
                results = {}

                def act():
                    time.sleep(0.1)
                    results["bob"] = httpx.post(cancel, headers=bob, timeout=5)
                    app.state.miniserve.begin_drain()
                    results["alice"] = httpx.post(cancel, headers=alice, timeout=5)

                t = threading.Thread(target=act)
                t.start()
                lines = [ln for ln in r.iter_lines() if ln.startswith("data: ")]
                t.join()
    assert results["bob"].status_code == 404  # not "forbidden": ids are not disclosed
    assert results["alice"].status_code == 200
    assert '"finish_reason": "cancelled"' in lines[-2]


def test_overload_returns_503_over_http():
    backend = ToyBackend(**SLOW)
    with live_server(backend, max_waiting=0, max_num_seqs=1) as (url, _):
        results = {}

        def long_request():
            results["long"] = httpx.post(
                f"{url}/v1/completions", json={"prompt": "x" * 50}, timeout=10
            )

        t = threading.Thread(target=long_request)
        t.start()
        time.sleep(0.15)
        r = httpx.post(f"{url}/v1/completions", json={"prompt": "hi"}, timeout=10)
        t.join()
        assert r.status_code == 503 and r.headers["retry-after"] == "1"
        assert r.json()["error"]["type"] == "overloaded_error"
        assert results["long"].status_code == 200
