"""``miniserve-bench``: run scenarios, run A/B suites, render reports.

miniserve-bench run --base-url http://127.0.0.1:8000 --scenario bench/scenarios/x.json
miniserve-bench suite bench/suites/batching.json --out results/batching
miniserve-bench report results/batching --charts docs/img/bench
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="miniserve-bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one scenario against a live server")
    r.add_argument("--base-url", default="http://127.0.0.1:8000")
    r.add_argument("--scenario", required=True)
    r.add_argument("--tokenizer", default="bytes", help="bytes | hf:<model_id>[@rev]")
    r.add_argument("--model", default=None, help="model name to send (external servers)")
    r.add_argument("--kind", default="miniserve", choices=["miniserve", "vllm", "openai"])
    r.add_argument("--out", default=None, help="write the full result JSON here")

    s = sub.add_parser("suite", help="start server configs and run scenarios, A/B rounds")
    s.add_argument("suite")
    s.add_argument("--out", default=None)
    s.add_argument("--rounds", type=int, default=None)

    rep = sub.add_parser("report", help="render Markdown (+charts) from a suite result dir")
    rep.add_argument("result_dir")
    rep.add_argument("--charts", default=None, help="directory for PNG charts")
    rep.add_argument("--out", default=None)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.cmd == "run":
        from miniserve.bench.runner import run_scenario
        from miniserve.bench.workload import PromptFactory, Scenario

        res = asyncio.run(
            run_scenario(
                args.base_url,
                Scenario.load(args.scenario),
                PromptFactory(args.tokenizer),
                args.model,
                args.kind,
            )
        )
        if args.out:
            Path(args.out).write_text(json.dumps(res))
        json.dump(
            {
                "summary": res["summary"],
                "server": {k: v for k, v in res["server"].items() if k != "series"},
            },
            sys.stdout,
            indent=2,
        )
        print()
    elif args.cmd == "suite":
        from miniserve.bench.suite import Suite, run_suite

        suite = Suite.load(args.suite)
        if args.rounds:
            suite.rounds = args.rounds
        out = Path(args.out or f"results/{suite.name}/{time.strftime('%Y%m%d-%H%M%S')}")
        run_suite(suite, out)
        print(out)
    else:
        from miniserve.bench.report import render

        md = render(Path(args.result_dir), Path(args.charts) if args.charts else None)
        if args.out:
            Path(args.out).write_text(md)
        else:
            print(md)


if __name__ == "__main__":
    main()
