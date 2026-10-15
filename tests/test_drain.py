"""Graceful shutdown with a real process and a real SIGTERM (the CLI entry point, not the app)."""

import json
import os
import signal
import socket
import subprocess
import sys
import time

import httpx


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start(port: int, **env) -> subprocess.Popen:
    base = {k: v for k, v in os.environ.items() if not k.startswith("MINISERVE_")}
    base.update(
        {
            "MINISERVE_BACKEND": "toy",
            "MINISERVE_PORT": str(port),
            "MINISERVE_WARMUP": "false",
            "MINISERVE_LOG_FORMAT": "json",
        }
    )
    base.update(env)
    proc = subprocess.Popen(
        [sys.executable, "-m", "miniserve"],
        env=base,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/readyz", timeout=1).status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        assert proc.poll() is None, proc.stdout.read()
        time.sleep(0.1)
    proc.kill()
    raise AssertionError("server did not become ready")


def test_sigterm_lets_in_flight_stream_finish_then_exits_cleanly():
    port = _free_port()
    proc = _start(port, MINISERVE_TOY_DECODE_MS_PER_TOKEN="20", MINISERVE_DRAIN_TIMEOUT_S="20")
    try:
        body = {"prompt": "x" * 60, "max_tokens": 60, "stream": True}  # ~1.2 s of decoding
        with httpx.stream(
            "POST", f"http://127.0.0.1:{port}/v1/completions", json=body, timeout=30
        ) as r:
            lines = r.iter_lines()
            first = next(ln for ln in lines if ln.startswith("data: "))
            assert '"text"' in first
            proc.send_signal(signal.SIGTERM)  # mid-stream
            rest = [ln for ln in lines if ln.startswith("data: ")]
        assert rest[-1] == "data: [DONE]"
        final = json.loads(rest[-2][6:])
        assert final["choices"][0]["finish_reason"] == "length"  # ran to max_tokens, not cancelled
        assert final["usage"]["completion_tokens"] == 60
        # uvicorn re-raises the captured SIGTERM after a complete graceful shutdown, so the
        # process ends "killed by SIGTERM" (143 in a container) rather than with code 0
        assert proc.wait(timeout=30) in (0, -signal.SIGTERM)
        logs = [json.loads(ln) for ln in proc.stdout.read().splitlines() if ln.startswith("{")]
        assert any(rec.get("msg") == "draining" for rec in logs)
        finished = [rec for rec in logs if rec.get("msg") == "request finished"]
        assert finished and finished[-1]["reason"] == "length"
    finally:
        if proc.poll() is None:
            proc.kill()


def test_refuses_public_bind_without_key():
    port = _free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith("MINISERVE_")}
    env.update(
        {"MINISERVE_BACKEND": "toy", "MINISERVE_HOST": "0.0.0.0", "MINISERVE_PORT": str(port)}
    )
    out = subprocess.run(
        [sys.executable, "-m", "miniserve"], env=env, capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 2 and "Refusing to bind" in out.stderr


def test_zero_drain_timeout_does_not_wait_for_long_streams():
    # Regression: drain_timeout_s=0 used to be passed to uvicorn as None ("wait forever").
    port = _free_port()
    proc = _start(port, MINISERVE_TOY_DECODE_MS_PER_TOKEN="20", MINISERVE_DRAIN_TIMEOUT_S="0")
    try:
        body = {"prompt": "y" * 500, "max_tokens": 500, "stream": True}  # ~10 s of decoding
        with httpx.stream(
            "POST", f"http://127.0.0.1:{port}/v1/completions", json=body, timeout=30
        ) as r:
            lines = r.iter_lines()
            next(ln for ln in lines if ln.startswith("data: "))
            t0 = time.monotonic()
            proc.send_signal(signal.SIGTERM)
            try:
                for _ in lines:
                    pass
            except httpx.HTTPError:
                pass  # the server may cut the connection
        proc.wait(timeout=15)
        assert time.monotonic() - t0 < 5
    finally:
        if proc.poll() is None:
            proc.kill()
