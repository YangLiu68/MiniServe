"""Benchmark suites: start each server configuration, run scenarios, repeat in rotating order.

A suite compares server *configurations* (environment overrides on top of shared defaults) on
the same *scenarios*. Each round runs every configuration once; the order rotates between rounds
(A B C, B C A, C A B) so slow drift in background load — this project was developed on a shared,
busy laptop — does not systematically favour whichever config runs first. Reports take the
median across rounds and show the spread.

External OpenAI-compatible servers (Ollama, vLLM) are configs with ``external`` set: the suite
does not start/stop them, it only sends the same workload.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import psutil

from miniserve import __version__
from miniserve.bench.runner import run_scenario
from miniserve.bench.workload import PromptFactory, Scenario

log = logging.getLogger(__name__)
RQ = Path.home() / ".resmgr" / "rq"


@dataclass
class ServerConfig:
    name: str
    env: dict[str, str] = field(default_factory=dict)
    external: dict[str, str] | None = None  # {"base_url", "model", "kind"}


@dataclass
class Suite:
    name: str
    configs: list[ServerConfig]
    scenarios: list[Scenario]
    base_env: dict[str, str] = field(default_factory=dict)
    rounds: int = 1
    tokenizer: str = "bytes"
    port: int = 8100
    description: str = ""
    # How the client talks to *managed* MiniServe servers. "miniserve" sends ignore_eos for
    # fixed-length outputs; "openai" sends plain OpenAI requests, for comparisons with servers
    # that cannot force output length (e.g. Ollama).
    client_kind: str = "miniserve"

    @classmethod
    def load(cls, path: str | Path) -> Suite:
        path = Path(path)
        d = json.loads(path.read_text())
        scenarios = []
        for s in d["scenarios"]:
            scenarios.append(
                Scenario.load(path.parent / s) if isinstance(s, str) else Scenario.from_dict(s)
            )
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            configs=[ServerConfig(**c) for c in d["configs"]],
            scenarios=scenarios,
            base_env={k: str(v) for k, v in d.get("base_env", {}).items()},
            rounds=d.get("rounds", 1),
            tokenizer=d.get("tokenizer", "bytes"),
            port=d.get("port", 8100),
            client_kind=d.get("client_kind", "miniserve"),
        )


def _run(cmd: list[str], timeout: float = 10) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def collect_env() -> dict[str, Any]:
    info: dict[str, Any] = {
        "miniserve": __version__,
        "git_sha": _run(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_run(["git", "status", "--porcelain", "--untracked-files=no"])),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor(),
        "cpu_count": os.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "loadavg": [round(x, 2) for x in os.getloadavg()],
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    for mod in ("torch", "transformers"):
        try:
            info[mod] = __import__(mod).__version__
        except ImportError:
            pass
    if RQ.exists():  # shared-machine resource manager: record contention alongside results
        status = _run([str(RQ), "status"]) or ""
        # Machine-level lines only: other sessions' jobs and account details stay out of results.
        keep = ("machine:", "disk:", "thermal:")
        machine = [ln for ln in status.splitlines() if ln.startswith(keep)]
        info["rq_status"] = "\n".join(machine) or None
    return info


class ManagedServer:
    """A MiniServe subprocess we started (and therefore may signal by PID)."""

    def __init__(self, env: dict[str, str], port: int, log_path: Path) -> None:
        clean = {k: v for k, v in os.environ.items() if not k.startswith("MINISERVE_")}
        clean.update(
            {
                "MINISERVE_HOST": "127.0.0.1",
                "MINISERVE_PORT": str(port),
                "MINISERVE_LOG_FORMAT": "json",
                "MINISERVE_RATE_LIMIT_RPM": "0",
            }
        )
        clean.update(env)
        self.port = port
        self.log_path = log_path
        self._log = open(log_path, "w")  # noqa: SIM115 - closed in stop()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "miniserve"],
            env=clean,
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def wait_ready(self, timeout: float = 900) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited ({self.proc.returncode}); see {self.log_path}")
            try:
                r = httpx.get(f"{self.base_url}/readyz", timeout=2)
                if r.status_code == 200:
                    return r.json()
                if r.json().get("status") == "failed":
                    raise RuntimeError(f"server failed to load: {r.json().get('error')}")
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        raise TimeoutError(f"server not ready after {timeout}s; see {self.log_path}")

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self._log.close()


def run_suite(suite: Suite, out_dir: Path) -> Path:
    raw = out_dir / "raw"
    logs = out_dir / "logs"
    raw.mkdir(parents=True, exist_ok=True)
    logs.mkdir(exist_ok=True)
    factory = PromptFactory(suite.tokenizer)
    manifest = {
        "suite": suite.name,
        "description": suite.description,
        "env": collect_env(),
        "configs": [vars(c) for c in suite.configs],
        "rounds": suite.rounds,
        "scenarios": [s.name for s in suite.scenarios],
        "runs": [],
    }
    n = len(suite.configs)
    for rnd in range(suite.rounds):
        order = suite.configs[rnd % n :] + suite.configs[: rnd % n]
        for cfg in order:
            server = None
            try:
                if cfg.external:
                    base_url, model = cfg.external["base_url"], cfg.external.get("model")
                    kind, ready = cfg.external.get("kind", "openai"), {}
                else:
                    env = {**suite.base_env, **cfg.env}
                    server = ManagedServer(env, suite.port, logs / f"{cfg.name}__r{rnd}.log")
                    log.info("starting %s (round %d)", cfg.name, rnd)
                    ready = server.wait_ready()
                    base_url, model, kind = server.base_url, None, suite.client_kind
                for scen in suite.scenarios:
                    log.info("  %s / %s", cfg.name, scen.name)
                    res = asyncio.run(run_scenario(base_url, scen, factory, model, kind))
                    res.update(
                        config=cfg.name,
                        round=rnd,
                        server_info=ready,
                        rq_status=collect_env().get("rq_status"),
                    )
                    path = raw / f"{scen.name}__{cfg.name}__r{rnd}.json"
                    path.write_text(json.dumps(res))
                    s = res["summary"]["all"]
                    log.info(
                        "    %s req ok, %.1f out tok/s, TTFT p50 %s ms",
                        s.get("completed"),
                        s.get("output_token_throughput", 0),
                        (s.get("ttft_ms") or {}).get("p50"),
                    )
                    manifest["runs"].append(path.name)
            finally:
                if server is not None:
                    server.stop()
    manifest["env_after"] = collect_env()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if shutil.which("git") is None:  # pragma: no cover
        pass
    return out_dir
