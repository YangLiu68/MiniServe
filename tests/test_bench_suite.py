"""Suite runner (managed server subprocesses, rotation, manifest) + report rendering; logging."""

import json
import logging

import pytest

from miniserve.bench.suite import Suite, run_suite
from miniserve.logging_setup import JsonFormatter, KeyValueFormatter


def test_toy_suite_end_to_end_and_report(tmp_path):
    pytest.importorskip("matplotlib")
    from miniserve.bench.report import render

    suite_file = tmp_path / "suite.json"
    suite_file.write_text(
        json.dumps(
            {
                "name": "tiny-suite",
                "rounds": 2,
                "port": 0,
                "base_env": {
                    "MINISERVE_BACKEND": "toy",
                    "MINISERVE_TOY_STEP_OVERHEAD_MS": "2",
                    "MINISERVE_WARMUP": "false",
                },
                "configs": [
                    {"name": "seq", "env": {"MINISERVE_MAX_NUM_SEQS": "1"}},
                    {"name": "batch", "env": {"MINISERVE_MAX_NUM_SEQS": "8"}},
                ],
                "scenarios": [
                    {
                        "name": f"s-c{c}",
                        "group": "s",
                        "x": c,
                        "streams": [
                            {
                                "mode": "closed",
                                "concurrency": c,
                                "num_requests": 4,
                                "prompt_tokens": 12,
                                "max_tokens": 5,
                            }
                        ],
                    }
                    for c in (1, 4)
                ],
            }
        )
    )
    suite = Suite.load(suite_file)
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        suite.port = s.getsockname()[1]
    out = run_suite(suite, tmp_path / "out")

    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["runs"]) == 2 * 2 * 2  # rounds x configs x scenarios
    assert manifest["env"]["cpu_count"] and "loadavg" in manifest["env"]
    run = json.loads((out / "raw" / "s-c4__batch__r1.json").read_text())
    assert run["summary"]["all"]["completed"] == 4
    assert run["server"]["mean_batch_seqs"] > 1  # batching actually happened
    # manifest["runs"] is in execution order: round 0 starts with "seq", round 1 rotates to "batch"
    first_config = [name.split("__")[1] for name in manifest["runs"][::2]]
    assert first_config == ["seq", "batch", "batch", "seq"]

    md = render(out, tmp_path / "charts")
    assert "| seq |" in md and "| batch |" in md and "**s-c4**" in md
    assert md.index("**s-c1**") < md.index("**s-c4**")  # suite order, not file order
    assert (tmp_path / "charts" / "s_output_token_throughput.png").stat().st_size > 1000


def _record(**extra):
    rec = logging.LogRecord("miniserve", logging.INFO, __file__, 1, "request finished", (), None)
    for k, v in extra.items():
        setattr(rec, k, v)
    return rec


def test_log_formatters_include_structured_fields():
    rec = _record(request_id="cmpl-1", reason="stop", ttft_ms=12.5)
    text = KeyValueFormatter().format(rec)
    assert "request finished" in text and "request_id=cmpl-1" in text and "ttft_ms=12.5" in text
    obj = json.loads(JsonFormatter().format(rec))
    assert obj["msg"] == "request finished" and obj["reason"] == "stop" and obj["level"] == "INFO"


def test_all_committed_suites_are_valid():
    from pathlib import Path

    from miniserve.config import Settings

    files = sorted(Path(__file__).parent.parent.joinpath("bench", "suites").glob("*.json"))
    assert len(files) >= 7
    for f in files:
        suite = Suite.load(f)
        assert suite.configs and suite.scenarios, f.name
        for cfg in suite.configs:
            env = {**suite.base_env, **cfg.env}
            fields = {
                k.removeprefix("MINISERVE_").lower(): v
                for k, v in env.items()
                if k.startswith("MINISERVE_")
            }
            Settings(_env_file=None, **fields)  # every MINISERVE_* override must be valid
            unknown = set(fields) - set(Settings.model_fields)
            assert not unknown, f"{f.name}/{cfg.name}: unknown settings {unknown}"
