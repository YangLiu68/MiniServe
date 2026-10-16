"""Aggregate suite results into Markdown tables (+ PNG charts for sweeps).

Every number in a table is the **median across rounds**; the headline columns also show the
min-max spread so a reader can see how noisy the run was.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

# Reference categorical palette (dataviz skill, light mode), fixed order — never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def _get(d: dict[str, Any], path: str) -> Any:
    for k in path.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _median(values: list[Any]) -> float | None:
    xs = [v for v in values if isinstance(v, int | float)]
    return statistics.median(xs) if xs else None


def load_runs(result_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((result_dir / "manifest.json").read_text())
    runs = [json.loads(p.read_text()) for p in sorted((result_dir / "raw").glob("*.json"))]
    return manifest, runs


def aggregate(runs: list[dict[str, Any]], scope: str = "all") -> dict[tuple[str, str], dict]:
    """(scenario, config) -> {metric_path: (median, min, max)} over rounds."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in runs:
        groups[(r["scenario"], r["config"])].append(r)
    metrics = [
        "summary.{s}.completed",
        "summary.{s}.errors",
        "summary.{s}.request_throughput",
        "summary.{s}.output_token_throughput",
        "summary.{s}.ttft_ms.p50",
        "summary.{s}.ttft_ms.p95",
        "summary.{s}.ttft_ms.p99",
        "summary.{s}.tpot_ms.p50",
        "summary.{s}.tpot_ms.p95",
        "summary.{s}.itl_ms.p95",
        "summary.{s}.itl_ms.p99",
        "summary.{s}.e2e_ms.p50",
        "summary.{s}.e2e_ms.p95",
        "summary.{s}.e2e_ms.p99",
        "summary.{s}.prefix_cached_fraction",
        "summary.{s}.mean_output_tokens",
        "server.peak_rss_mb",
        "server.peak_device_mb",
        "server.peak_kv_usage",
        "server.mean_batch_seqs",
        "server.mean_step_ms",
        "server.preemptions",
        "server.peak_waiting",
        "server.spec_acceptance",
        "server.spec_accepted_tokens",
    ]
    out = {}
    for key, rs in groups.items():
        agg = {"rounds": len(rs), "group": rs[0].get("group"), "x": rs[0].get("x")}
        for m in metrics:
            path = m.format(s=scope)
            vals = [_get(r, path) for r in rs]
            nums = [v for v in vals if isinstance(v, int | float)]
            agg[m.format(s="S")] = (
                _median(nums),
                min(nums) if nums else None,
                max(nums) if nums else None,
            )
        out[key] = agg
    return out


def _fmt(v: float | None, digits: int = 1) -> str:
    if v is None:
        return "–"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{digits}f}"


def _spread(t: tuple, digits: int = 1) -> str:
    med, lo, hi = t
    if med is None:
        return "–"
    if lo is None or hi is None or lo == hi:
        return _fmt(med, digits)
    return f"{_fmt(med, digits)} <sub>({_fmt(lo, digits)}–{_fmt(hi, digits)})</sub>"


COLUMNS = [
    ("ok/err", None),
    ("out tok/s", "summary.S.output_token_throughput"),
    ("req/s", "summary.S.request_throughput"),
    ("TTFT p50", "summary.S.ttft_ms.p50"),
    ("TTFT p95", "summary.S.ttft_ms.p95"),
    ("TPOT p50", "summary.S.tpot_ms.p50"),
    ("ITL p95", "summary.S.itl_ms.p95"),
    ("E2E p95", "summary.S.e2e_ms.p95"),
]
SERVER_COLUMNS = [
    ("batch", "server.mean_batch_seqs"),
    ("step ms", "server.mean_step_ms"),
    ("peak KV", "server.peak_kv_usage"),
    ("peak RSS MB", "server.peak_rss_mb"),
    ("preempt", "server.preemptions"),
    ("cached", "summary.S.prefix_cached_fraction"),
    ("draft accept", "server.spec_acceptance"),
]


def table(agg: dict, scenario: str, configs: list[str], with_server: bool = True) -> str:
    cols = COLUMNS + (SERVER_COLUMNS if with_server else [])
    rows = [agg[(scenario, c)] for c in configs if (scenario, c) in agg]
    if not any(r.get("server.spec_acceptance", (None,))[0] is not None for r in rows):
        cols = [c for c in cols if c[0] != "draft accept"]  # only when speculation was used
    lines = ["| config | " + " | ".join(c for c, _ in cols) + " |", "|---|" + "---:|" * len(cols)]
    for cfg in configs:
        a = agg.get((scenario, cfg))
        if a is None:
            continue
        cells = []
        for name, path in cols:
            if path is None:
                ok, err = a["summary.S.completed"][0], a["summary.S.errors"][0]
                cells.append(f"{_fmt(ok, 0)}/{_fmt(err, 0)}")
            elif name in ("out tok/s", "TTFT p50", "TTFT p95"):
                cells.append(_spread(a[path]))
            elif name in ("peak KV", "cached", "draft accept"):
                v = a[path][0]
                cells.append("–" if v is None else f"{v:.0%}")
            else:
                cells.append(_fmt(a[path][0], 2 if name == "req/s" else 1))
        lines.append(f"| {cfg} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9)


def sweep_charts(
    agg: dict, group: str, configs: list[str], out_dir: Path, xlabel: str
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("output_token_throughput", "summary.S.output_token_throughput", "Output tokens / s"),
        ("ttft_p95", "summary.S.ttft_ms.p95", "TTFT p95 (ms)"),
        ("tpot_p50", "summary.S.tpot_ms.p50", "Time per output token, p50 (ms)"),
    ]
    paths = []
    for slug, metric, ylabel in panels:
        fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        _style_axes(ax)
        ends = []
        for i, cfg in enumerate(configs):
            pts = sorted(
                (a["x"], a[metric][0])
                for (sc, c), a in agg.items()
                if c == cfg and a["group"] == group and a[metric][0] is not None
            )
            if not pts:
                continue
            xs, ys = zip(*pts, strict=True)
            color = SERIES[i % len(SERIES)]
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=2,
                solid_capstyle="round",
                solid_joinstyle="round",
                label=cfg,
                zorder=3,
            )
            ax.scatter(xs, ys, s=36, color=color, edgecolors=SURFACE, linewidths=2, zorder=4)
            ends.append(
                ax.annotate(
                    cfg,
                    (xs[-1], ys[-1]),
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                    color=INK_2,
                )
            )
        xs_all = sorted({a["x"] for a in agg.values() if a["group"] == group})
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs_all, [f"{x:g}" for x in xs_all])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ys_all = [v for line in ax.get_lines() for v in line.get_ydata() if v > 0]
        log_y = slug != "output_token_throughput" and ys_all and max(ys_all) / min(ys_all) > 30
        if log_y:  # latencies spanning orders of magnitude
            ax.set_yscale("log")
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
        ax.set_ylabel(ylabel + (" — log scale" if log_y else ""), color=INK_2, fontsize=9)
        ax.set_title(f"{group}: {ylabel}", color=INK, fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
        fig.tight_layout()
        # Converging lines: overlapping end labels are dropped (the legend carries identity)
        # rather than nudged away from the lines they name.
        ys_px = sorted(ax.transData.transform(a.xy)[1] for a in ends)
        if any(b - a < 14 for a, b in zip(ys_px, ys_px[1:], strict=False)):
            for a in ends:
                a.remove()
        path = out_dir / f"{group}_{slug}.png"
        fig.savefig(path, facecolor=SURFACE)
        plt.close(fig)
        paths.append(path)
    return paths


def render(result_dir: Path, charts_dir: Path | None = None, tenants: bool = True) -> str:
    manifest, runs = load_runs(result_dir)
    configs = [c["name"] for c in manifest["configs"]]
    present = {r["scenario"] for r in runs}
    scenarios = [s for s in manifest["scenarios"] if s in present]  # suite order, not file order
    agg = aggregate(runs)
    env = manifest["env"]
    out = [
        f"### Suite `{manifest['suite']}`",
        "",
        manifest.get("description", ""),
        "",
        f"- Rounds: {manifest['rounds']} (config order rotated each round; cells are the "
        "median across rounds, <sub>(min–max)</sub> where it varies)",
        f"- Host: {env.get('cpu')}, {env.get('memory_gb')} GB, {env.get('platform')}; "
        f"torch {env.get('torch')}, transformers {env.get('transformers')}",
        f"- Commit: `{(env.get('git_sha') or '')[:10]}`{' (dirty)' if env.get('git_dirty') else ''}"
        f"; load average at start {env.get('loadavg')}, at end "
        f"{manifest.get('env_after', {}).get('loadavg')}",
        "- Latencies in ms (client-side, streaming). TPOT = (E2E − TTFT)/(tokens − 1).",
        "",
    ]
    groups = defaultdict(list)
    for sc in scenarios:
        g = next((r.get("group") for r in runs if r["scenario"] == sc), None)
        groups[g].append(sc)
    for g, scs in groups.items():
        for sc in scs:
            out += [f"**{sc}**", "", table(agg, sc, configs), ""]
            if tenants:
                ts = sorted(
                    {t for r in runs if r["scenario"] == sc for t in r["summary"] if t != "all"}
                )
                for t in ts:
                    tagg = aggregate([r for r in runs if r["scenario"] == sc], scope=t)
                    out += [
                        f"<sub>tenant `{t}`</sub>",
                        "",
                        table(tagg, sc, configs, with_server=False),
                        "",
                    ]
        if g and charts_dir is not None:
            charts_dir.mkdir(parents=True, exist_ok=True)
            for p in sweep_charts(agg, g, configs, charts_dir, "client concurrency"):
                out += [f"![{p.stem}]({p.as_posix()})", ""]
    return "\n".join(out)
