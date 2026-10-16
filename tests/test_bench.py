"""Benchmark tooling: stats, workload construction, and an end-to-end run against a live server."""

import asyncio
import random

from miniserve.backends.toy import ToyBackend
from miniserve.bench.runner import parse_prometheus, run_scenario
from miniserve.bench.stats import percentile, summarize
from miniserve.bench.workload import PromptFactory, Scenario, build_requests

from conftest import live_server


def test_percentile_matches_numpy_linear_convention():
    xs = [5, 1, 4, 2, 3]
    assert percentile(xs, 0) == 1 and percentile(xs, 100) == 5 and percentile(xs, 50) == 3
    assert percentile([10, 20], 95) == 19.5
    assert percentile([], 50) is None


def test_summary_excludes_warmup_and_errors():
    base = dict(
        stream=0,
        tenant="t",
        prompt_tokens=10,
        cached_tokens=0,
        itl=[0.01, 0.02],
        status=200,
        server=None,
    )
    rs = [
        dict(
            base, start=0.0, output_tokens=3, ttft=0.1, e2e=0.2, finish_reason="length", error=None
        ),
        dict(
            base, start=1.0, output_tokens=3, ttft=0.3, e2e=0.5, finish_reason="length", error=None
        ),
        dict(
            base,
            start=1.5,
            output_tokens=0,
            ttft=None,
            e2e=0.1,
            finish_reason=None,
            error="HTTP 503",
        ),
    ]
    s = summarize(rs, warmup_s=0.5)
    assert s["requests"] == 2 and s["completed"] == 1 and s["errors"] == 1
    assert s["ttft_ms"]["p50"] == 300.0
    assert s["tpot_ms"]["p50"] == 100.0  # (0.5 - 0.3) / (3 - 1)


def test_byte_prompts_have_exact_lengths_and_unique_heads():
    f = PromptFactory("bytes")
    rng = random.Random(0)
    prompts = [f.make(50, rng) for _ in range(20)]
    assert all(f.count(p) == 50 for p in prompts)
    assert len({p[:12] for p in prompts}) == 20


def test_build_requests_is_deterministic_and_shares_prefix():
    sc = Scenario.from_dict(
        {
            "name": "x",
            "seed": 3,
            "streams": [
                {
                    "mode": "poisson",
                    "rate": 5,
                    "num_requests": 10,
                    "prompt_tokens": 40,
                    "max_tokens": {"uniform": [4, 8]},
                    "shared_prefix_tokens": 20,
                }
            ],
        }
    )
    a = build_requests(sc, PromptFactory("bytes"))[0]
    b = build_requests(sc, PromptFactory("bytes"))[0]
    assert [r.prompt for r in a] == [r.prompt for r in b]
    assert all(r.prompt[:20] == a[0].prompt[:20] for r in a)
    assert all(4 <= r.max_tokens <= 8 for r in a)
    arrivals = [r.arrival for r in a]
    assert arrivals == sorted(arrivals) and arrivals[0] > 0


def test_parse_prometheus():
    m = parse_prometheus('# HELP x\nfoo_total 3.0\nbar{kind="a"} 1e3\n')
    assert m == {"foo_total": 3.0, 'bar{kind="a"}': 1000.0}


def test_end_to_end_scenario_against_live_server():
    sc = Scenario.from_dict(
        {
            "name": "e2e",
            "streams": [
                {
                    "tenant": "a",
                    "mode": "closed",
                    "concurrency": 3,
                    "num_requests": 9,
                    "prompt_tokens": 20,
                    "max_tokens": 6,
                },
                {
                    "tenant": "b",
                    "mode": "poisson",
                    "rate": 50,
                    "num_requests": 5,
                    "prompt_tokens": 10,
                    "max_tokens": 4,
                },
            ],
        }
    )
    with live_server(ToyBackend(step_overhead_ms=2)) as (url, _):
        res = asyncio.run(run_scenario(url, sc, PromptFactory("bytes")))
    s = res["summary"]
    assert s["all"]["completed"] == 14 and s["all"]["errors"] == 0
    assert s["a"]["mean_output_tokens"] == 6.0  # ignore_eos: exact lengths
    assert s["b"]["mean_prompt_tokens"] == 10.0
    assert s["all"]["ttft_ms"]["p50"] > 0
    assert res["server"]["steps"] > 0 and res["server"]["mean_batch_seqs"] >= 1


def test_prompt_datasets_and_ignore_eos_switch(tmp_path):
    from miniserve.bench.client import build_payload

    data = tmp_path / "p.json"
    data.write_text('{"set": ["first prompt", "second prompt"]}')
    sc = Scenario.from_dict(
        {
            "name": "d",
            "streams": [
                {"num_requests": 5, "prompts": f"{data}:set", "endpoint": "chat"},
                {"num_requests": 1, "prompts": ["only"], "ignore_eos": False},
            ],
        }
    )
    a, b = build_requests(sc, PromptFactory("bytes"))
    assert sorted(r.prompt for r in a[:2]) == ["first prompt", "second prompt"]  # each pass
    assert {r.prompt for r in a} == {"first prompt", "second prompt"}  # covers the dataset
    chat = build_payload(a[0], None, "miniserve")
    assert chat["messages"][0]["content"] in ("first prompt", "second prompt")
    assert chat["ignore_eos"] is True  # default: fixed-length outputs
    assert "ignore_eos" not in build_payload(b[0], None, "miniserve")  # natural endings
