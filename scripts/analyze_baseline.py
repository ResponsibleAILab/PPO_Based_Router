#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Config
# ============================================================

ALL_METHODS = ["epsilon", "ppo", "softmax", "ucb", "thompson", "moucb"]
TASK_GROUPS = ["code", "explain", "tests"]

METHOD_COLORS = {
    "epsilon": "#ff7f0e",
    "ppo": "#1f77b4",
    "softmax": "#9467bd",
    "ucb": "#2ca02c",
    "thompson": "#b8860b",
    "moucb": "#d62728",
}

METHOD_HATCHES = {
    "epsilon": "//",
    "ppo": "\\\\",
    "softmax": "xx",
    "ucb": "-",
    "thompson": "++",
    "moucb": "oo",
}

METHOD_MARKERS = {
    "epsilon": "s",
    "ppo": "o",
    "softmax": "^",
    "ucb": "D",
    "thompson": "X",
    "moucb": "P",
}

GRID_KW = dict(linestyle="--", alpha=0.35, linewidth=0.8)
BAR_KW = dict(edgecolor="black", linewidth=0.7)

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})


# ============================================================
# Helpers
# ============================================================

def safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def is_valid_number(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def pct(vals, q):
    clean = [float(v) for v in vals if is_valid_number(v)]
    if not clean:
        return math.nan
    return float(np.percentile(np.asarray(clean, dtype=float), q))


def fmt(x, digits=3):
    if not is_valid_number(x):
        return "–"
    return f"{float(x):.{digits}f}"


def canonical_method_name(text: str) -> str:
    s = (text or "").lower()
    if "epsilon" in s:
        return "epsilon"
    if "softmax" in s:
        return "softmax"
    if "thompson" in s:
        return "thompson"
    if "moucb" in s or "mo_ucb" in s or "vector_ucb" in s or "multiobjectiveucb" in s:
        return "moucb"
    if s == "ucb" or re.search(r"(^|_)ucb($|_)", s):
        return "ucb"
    if "ppo" in s:
        return "ppo"
    return "unknown"


def infer_task_group_from_filename(path: str) -> str:
    """
    Use ONLY the filename string to infer the task group.
    We intentionally ignore row-level tags.
    """
    base = os.path.basename(path).lower()

    # Strict filename pattern matching.
    if re.search(r"(^|_)code(_|\.|$)", base):
        return "code"
    if re.search(r"(^|_)explain(_|\.|$)", base):
        return "explain"
    if re.search(r"(^|_)tests(_|\.|$)", base) or re.search(r"(^|_)test(_|\.|$)", base):
        return "tests"

    return "unknown"


def infer_run_id(path: str) -> str:
    base = os.path.basename(path)
    stem = base[:-6] if base.endswith(".jsonl") else base
    m = re.search(r"_(\d{8}_\d{6})$", stem)
    if m:
        return m.group(1)
    return stem


def method_from_filename(path: str) -> str:
    return canonical_method_name(os.path.basename(path))


def color_for(method: str) -> str:
    return METHOD_COLORS.get(method, "#bbbbbb")


def hatch_for(method: str) -> str | None:
    return METHOD_HATCHES.get(method)


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def group_by(records, keys):
    out = defaultdict(list)
    for r in records:
        out[tuple(r.get(k) for k in keys)].append(r)
    return out


# ============================================================
# Log discovery
# ============================================================

def find_logs(logdir: str):
    """
    Finds all route log jsonl files. Keeps things flexible enough for:
      route_log_ppo_code_2025....jsonl
      route_log_ucb_explain_....jsonl
      route_log_thompson_tests_....jsonl
      route_... style variants
    """
    patterns = [
        os.path.join(logdir, "*.jsonl"),
        os.path.join(logdir, "**", "*.jsonl"),
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))

    selected = []
    for path in files:
        name = os.path.basename(path).lower()
        if "route" not in name:
            continue
        if canonical_method_name(name) == "unknown":
            continue
        task_group = infer_task_group_from_filename(name)
        if task_group not in TASK_GROUPS:
            continue
        selected.append(path)

    return sorted(set(selected))


# ============================================================
# Record collection
# ============================================================

def normalize_latency_seconds(row: dict) -> float | None:
    """
    Prefer explicit seconds fields first.
    Only convert ms fields when those fields are actually used.
    """
    if row.get("latency_s") is not None:
        return safe_float(row.get("latency_s"))

    if row.get("latency") is not None:
        return safe_float(row.get("latency"))

    if row.get("exec_ms") is not None:
        val = safe_float(row.get("exec_ms"))
        return None if val is None else val / 1000.0

    if row.get("latency_ms") is not None:
        val = safe_float(row.get("latency_ms"))
        return None if val is None else val / 1000.0

    return None


def collect_records(files):
    records = []

    for path in files:
        method = method_from_filename(path)
        task_group = infer_task_group_from_filename(path)
        run_id = infer_run_id(path)

        for row in load_jsonl(path):
            backend = row.get("backend")
            if backend is None:
                backend = row.get("backend_id")
            if backend is None:
                backend = row.get("selected_backend")

            rec = {
                "__file__": path,
                "__filename__": os.path.basename(path),
                "__method__": method,
                "__task_group__": task_group,
                "__run_id__": run_id,
                "latency_s": normalize_latency_seconds(row),
                "reward": safe_float(row.get("reward", row.get("score"))),
                "quality": safe_float(
                    row.get("quality",
                    row.get("quality_score",
                    row.get("structural_quality")))
                ),
                "cost": safe_float(
                    row.get("cost",
                    row.get("cost_estimate",
                    row.get("token_cost")))
                ),
                "backend": None if backend is None else str(backend),
                "alpha_quality": safe_float(row.get("alpha_quality")),
                "beta_latency": safe_float(row.get("beta_latency")),
                "gamma_cost": safe_float(row.get("gamma_cost")),
            }
            records.append(rec)

    return records


# ============================================================
# Summaries
# ============================================================

def summarize(records):
    lat = [r["latency_s"] for r in records if is_valid_number(r.get("latency_s"))]
    rew = [r["reward"] for r in records if is_valid_number(r.get("reward"))]
    qual = [r["quality"] for r in records if is_valid_number(r.get("quality"))]
    cost = [r["cost"] for r in records if is_valid_number(r.get("cost"))]
    backends = [r["backend"] for r in records if r.get("backend") is not None]

    backend_counts = {}
    for b in sorted(set(backends)):
        backend_counts[b] = backends.count(b)

    return {
        "n": len(records),
        "mean_latency": mean(lat) if lat else math.nan,
        "p50_latency": pct(lat, 50),
        "p95_latency": pct(lat, 95),
        "mean_reward": mean(rew) if rew else math.nan,
        "mean_quality": mean(qual) if qual else math.nan,
        "mean_cost": mean(cost) if cost else math.nan,
        "backend_counts": backend_counts,
    }


def build_summary_rows(records):
    grouped = group_by(records, ["__task_group__", "__method__"])
    rows = []

    for task_group in TASK_GROUPS:
        for method in ALL_METHODS:
            recs = grouped.get((task_group, method), [])
            s = summarize(recs)
            rows.append({
                "task_group": task_group,
                "method": method,
                "n": s["n"],
                "mean_latency": s["mean_latency"],
                "p50_latency": s["p50_latency"],
                "p95_latency": s["p95_latency"],
                "mean_reward": s["mean_reward"],
                "mean_quality": s["mean_quality"],
                "mean_cost": s["mean_cost"],
                "backend_counts": s["backend_counts"],
            })
    return rows


def build_backend_rows(records):
    grouped = group_by(records, ["__task_group__", "__method__"])
    rows = []

    for task_group in TASK_GROUPS:
        for method in ALL_METHODS:
            recs = grouped.get((task_group, method), [])
            s = summarize(recs)
            for backend, count in s["backend_counts"].items():
                rows.append({
                    "task_group": task_group,
                    "method": method,
                    "backend": backend,
                    "count": count,
                })
    return rows


# ============================================================
# CSV output
# ============================================================

def save_csv(rows, path, fieldnames=None):
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ============================================================
# Plotting
# ============================================================

def plot_per_request_pareto_all(records, task_group, outdir):
    """
    Single-topic figure:
    x-axis = latency
    y-axis = reward
    one scatter cloud per method
    filtered to one filename-derived category
    """
    task_records = [r for r in records if r["__task_group__"] == task_group]
    if not task_records:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for method in ALL_METHODS:
        xs = []
        ys = []
        for r in task_records:
            if r["__method__"] != method:
                continue
            lat = r.get("latency_s")
            rew = r.get("reward")
            if not is_valid_number(lat) or not is_valid_number(rew):
                continue
            xs.append(lat)
            ys.append(rew)

        if not xs:
            continue

        ax.scatter(
            xs,
            ys,
            s=18,
            alpha=0.8,
            label=method.upper(),
            color=color_for(method),
            marker=METHOD_MARKERS.get(method, "o"),
            edgecolors="none",
        )

    ax.set_xlabel("Latency (s) ↓", fontweight="bold")
    ax.set_ylabel("Reward ↑", fontweight="bold")
    ax.set_title(f"{task_group.capitalize()} Prompts: Per-Request Pareto (Reward vs Latency)")
    ax.grid(True, **GRID_KW)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{task_group}_per_request_pareto_reward_vs_latency.pdf"), bbox_inches="tight")
    plt.close(fig)

def _barplot_metric_for_task(summary_rows, task_group, metric_key, ylabel, title, outpath):
    rows = [r for r in summary_rows if r["task_group"] == task_group]
    methods = [m for m in ALL_METHODS if any(r["method"] == m for r in rows)]

    vals = []
    present_methods = []
    for m in methods:
        row = next((r for r in rows if r["method"] == m), None)
        if row is None:
            continue
        vals.append(row[metric_key])
        present_methods.append(m)

    if not present_methods:
        return

    x = np.arange(len(present_methods))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    bars = ax.bar(
        x,
        vals,
        color=[color_for(m) for m in present_methods],
        **BAR_KW,
    )

    for bar, method in zip(bars, present_methods):
        hatch = hatch_for(method)
        if hatch:
            bar.set_hatch(hatch)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in present_methods], rotation=20, ha="right")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_xlabel("Method", fontweight="bold")
    ax.set_title(title)
    ax.grid(True, axis="y", **GRID_KW)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_quality_by_task(summary_rows, task_group, outdir):
    _barplot_metric_for_task(
        summary_rows=summary_rows,
        task_group=task_group,
        metric_key="mean_quality",
        ylabel="Mean Quality ↑",
        title=f"{task_group.capitalize()} Prompts: Mean Quality by Method",
        outpath=os.path.join(outdir, f"{task_group}_quality_by_method.pdf"),
    )


def plot_latency_by_task(summary_rows, task_group, outdir):
    _barplot_metric_for_task(
        summary_rows=summary_rows,
        task_group=task_group,
        metric_key="mean_latency",
        ylabel="Mean Latency (s) ↓",
        title=f"{task_group.capitalize()} Prompts: Mean Latency by Method",
        outpath=os.path.join(outdir, f"{task_group}_latency_by_method.pdf"),
    )


def plot_reward_by_task(summary_rows, task_group, outdir):
    _barplot_metric_for_task(
        summary_rows=summary_rows,
        task_group=task_group,
        metric_key="mean_reward",
        ylabel="Mean Reward ↑",
        title=f"{task_group.capitalize()} Prompts: Mean Reward by Method",
        outpath=os.path.join(outdir, f"{task_group}_reward_by_method.pdf"),
    )


def plot_cost_by_task(summary_rows, task_group, outdir):
    _barplot_metric_for_task(
        summary_rows=summary_rows,
        task_group=task_group,
        metric_key="mean_cost",
        ylabel="Mean Cost ↓",
        title=f"{task_group.capitalize()} Prompts: Mean Cost by Method",
        outpath=os.path.join(outdir, f"{task_group}_cost_by_method.pdf"),
    )


def plot_backend_usage_for_task(backend_rows, task_group, outdir):
    rows = [r for r in backend_rows if r["task_group"] == task_group]
    if not rows:
        return

    methods = [m for m in ALL_METHODS if any(r["method"] == m for r in rows)]
    backends = sorted(set(r["backend"] for r in rows))

    if not methods or not backends:
        return

    x = np.arange(len(backends))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    for i, method in enumerate(methods):
        counts = []
        for backend in backends:
            row = next(
                (r for r in rows if r["method"] == method and r["backend"] == backend),
                None
            )
            counts.append(row["count"] if row else 0)

        bars = ax.bar(
            x + (i - (len(methods) - 1) / 2) * width,
            counts,
            width,
            label=method.upper(),
            color=color_for(method),
            **BAR_KW,
        )

        hatch = hatch_for(method)
        if hatch:
            for bar in bars:
                bar.set_hatch(hatch)

    ax.set_xticks(x)
    ax.set_xticklabels(backends, rotation=20, ha="right")
    ax.set_xlabel("Backend / Model", fontweight="bold")
    ax.set_ylabel("Selection Count", fontweight="bold")
    ax.set_title(f"{task_group.capitalize()} Prompts: Backend Usage by Method")
    ax.grid(True, axis="y", **GRID_KW)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{task_group}_backend_usage_by_method.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_metric_distribution_for_task(records, task_group, metric_key, ylabel, title, outpath):
    task_records = [r for r in records if r["__task_group__"] == task_group]
    if not task_records:
        return

    data = {}
    for method in ALL_METHODS:
        vals = [
            r[metric_key]
            for r in task_records
            if r["__method__"] == method and is_valid_number(r.get(metric_key))
        ]
        if vals:
            data[method] = vals

    methods = [m for m in ALL_METHODS if m in data]
    if not methods:
        return

    pos = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    bp = ax.boxplot(
        [data[m] for m in methods],
        positions=pos,
        widths=0.55,
        showfliers=False,
        patch_artist=True,
    )

    for box, method in zip(bp["boxes"], methods):
        box.set_facecolor(color_for(method))
        box.set_alpha(0.85)
        box.set_edgecolor("black")
        box.set_linewidth(0.7)
        hatch = hatch_for(method)
        if hatch:
            box.set_hatch(hatch)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.8)

    ax.set_xticks(pos)
    ax.set_xticklabels([m.upper() for m in methods], rotation=20, ha="right")
    ax.set_xlabel("Method", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title)
    ax.grid(True, axis="y", **GRID_KW)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# LaTeX paper table
# ============================================================

def write_paper_table(summary_rows, out_md, out_tex):
    """
    Compact combined table:
    Category | Method | n | Mean Lat. | p95 | Mean Rew. | Mean Qual. | Mean Cost
    """

    # Markdown
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Combined Results Table\n\n")
        f.write("| Category | Method | n | Mean Lat. | p95 | Mean Rew. | Mean Qual. | Mean Cost |\n")
        f.write("|:--|:--|--:|--:|--:|--:|--:|--:|\n")
        for task_group in TASK_GROUPS:
            rows = [r for r in summary_rows if r["task_group"] == task_group]
            for i, row in enumerate(rows):
                category_label = task_group if i == 0 else ""
                f.write(
                    f"| {category_label} | {row['method']} | {row['n']} | "
                    f"{fmt(row['mean_latency'])} | {fmt(row['p95_latency'])} | "
                    f"{fmt(row['mean_reward'])} | {fmt(row['mean_quality'])} | "
                    f"{fmt(row['mean_cost'])} |\n"
                )

    # LaTeX
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Category & Method & $n$ & Mean Lat. & p95 & Mean Rew. & Mean Qual. & Mean Cost \\",
        r"\midrule",
    ]

    for task_group in TASK_GROUPS:
        rows = [r for r in summary_rows if r["task_group"] == task_group]
        for i, row in enumerate(rows):
            category_label = task_group if i == 0 else ""
            tex_lines.append(
                f"{category_label} & {row['method']} & {row['n']} & "
                f"{fmt(row['mean_latency'])} & {fmt(row['p95_latency'])} & "
                f"{fmt(row['mean_reward'])} & {fmt(row['mean_quality'])} & "
                f"{fmt(row['mean_cost'])} \\\\"
            )
        if task_group != TASK_GROUPS[-1]:
            tex_lines.append(r"\midrule")

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Baseline routing results grouped by filename-derived category. Lower latency and cost are better. Higher reward and quality are better.}",
        r"\label{tab:baseline_combined_results}",
        r"\end{table}",
        "",
    ])

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines))


# ============================================================
# Reward config report
# ============================================================

def reward_configs_by_method(records):
    configs = defaultdict(set)
    for r in records:
        method = r["__method__"]
        a = safe_float(r.get("alpha_quality"))
        b = safe_float(r.get("beta_latency"))
        g = safe_float(r.get("gamma_cost"))
        if None in (a, b, g):
            continue
        configs[method].add((round(a, 10), round(b, 10), round(g, 10)))
    return dict(configs)


def write_reward_config_report(records, out_csv):
    configs = reward_configs_by_method(records)
    rows = []

    for method in ALL_METHODS:
        cfgs = sorted(configs.get(method, set()))
        rows.append({
            "method": method,
            "configs_seen": "; ".join(str(c) for c in cfgs) if cfgs else "missing",
        })

    save_csv(rows, out_csv, fieldnames=["method", "configs_seen"])


# ============================================================
# Console summary
# ============================================================

def print_summary(summary_rows):
    print("\n=== Baseline Summary by Filename-Derived Category ===")
    for task_group in TASK_GROUPS:
        print(f"\n[{task_group.upper()}]")
        rows = [r for r in summary_rows if r["task_group"] == task_group]
        for row in rows:
            print(
                f"  {row['method']:9s} "
                f"n={row['n']:4d}  "
                f"lat={fmt(row['mean_latency']):>7s}  "
                f"p95={fmt(row['p95_latency']):>7s}  "
                f"rew={fmt(row['mean_reward']):>7s}  "
                f"qual={fmt(row['mean_quality']):>7s}  "
                f"cost={fmt(row['mean_cost']):>7s}"
            )


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Analyze baseline router logs using filename-derived categories only."
    )
    ap.add_argument("--logdir", default="logs", help="Directory containing baseline JSONL logs.")
    ap.add_argument("--outdir", default="results_baseline", help="Directory for output files.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    files = find_logs(args.logdir)
    if not files:
        print(f"No matching JSONL files found in: {args.logdir}")
        raise SystemExit(1)

    print("Using files:")
    for f in files:
        print(" -", os.path.basename(f))

    records = collect_records(files)
    if not records:
        print("No records loaded.")
        raise SystemExit(2)

    summary_rows = build_summary_rows(records)
    backend_rows = build_backend_rows(records)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV outputs
    save_csv(
        summary_rows,
        os.path.join(args.outdir, f"summary_by_category_method_{timestamp}.csv"),
        fieldnames=[
            "task_group",
            "method",
            "n",
            "mean_latency",
            "p50_latency",
            "p95_latency",
            "mean_reward",
            "mean_quality",
            "mean_cost",
            "backend_counts",
        ],
    )

    save_csv(
        backend_rows,
        os.path.join(args.outdir, f"backend_usage_by_category_method_{timestamp}.csv"),
        fieldnames=["task_group", "method", "backend", "count"],
    )

    write_reward_config_report(
        records,
        os.path.join(args.outdir, f"reward_config_report_{timestamp}.csv"),
    )

    # Single-topic plots per category
    for task_group in TASK_GROUPS:
        plot_quality_by_task(summary_rows, task_group, args.outdir)
        plot_latency_by_task(summary_rows, task_group, args.outdir)
        plot_reward_by_task(summary_rows, task_group, args.outdir)
        plot_cost_by_task(summary_rows, task_group, args.outdir)
        plot_backend_usage_for_task(backend_rows, task_group, args.outdir)

        plot_metric_distribution_for_task(
            records=records,
            task_group=task_group,
            metric_key="quality",
            ylabel="Quality ↑",
            title=f"{task_group.capitalize()} Prompts: Quality Distribution by Method",
            outpath=os.path.join(args.outdir, f"{task_group}_quality_distribution.pdf"),
        )

        plot_metric_distribution_for_task(
            records=records,
            task_group=task_group,
            metric_key="latency_s",
            ylabel="Latency (s) ↓",
            title=f"{task_group.capitalize()} Prompts: Latency Distribution by Method",
            outpath=os.path.join(args.outdir, f"{task_group}_latency_distribution.pdf"),
        )

        plot_metric_distribution_for_task(
            records=records,
            task_group=task_group,
            metric_key="reward",
            ylabel="Reward ↑",
            title=f"{task_group.capitalize()} Prompts: Reward Distribution by Method",
            outpath=os.path.join(args.outdir, f"{task_group}_reward_distribution.pdf"),
        )

        plot_per_request_pareto_all(records, task_group, args.outdir)

    # Paper tables
    write_paper_table(
        summary_rows,
        os.path.join(args.outdir, "baseline_combined_table.md"),
        os.path.join(args.outdir, "baseline_combined_table.tex"),
    )

    print_summary(summary_rows)

    print(f"\nWrote outputs to: {args.outdir}")
    print("Generated:")
    print(" - summary_by_category_method_<timestamp>.csv")
    print(" - backend_usage_by_category_method_<timestamp>.csv")
    print(" - reward_config_report_<timestamp>.csv")
    print(" - baseline_combined_table.md")
    print(" - baseline_combined_table.tex")
    print(" - code/explain/tests quality_by_method.pdf")
    print(" - code/explain/tests latency_by_method.pdf")
    print(" - code/explain/tests reward_by_method.pdf")
    print(" - code/explain/tests cost_by_method.pdf")
    print(" - code/explain/tests backend_usage_by_method.pdf")
    print(" - code/explain/tests quality_distribution.pdf")
    print(" - code/explain/tests latency_distribution.pdf")
    print(" - code/explain/tests reward_distribution.pdf")


if __name__ == "__main__":
    main()