#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

ALL_METHODS = ["epsilon", "ppo", "softmax", "ucb", "thompson", "moucb"]
BASELINES = ["epsilon", "softmax", "ucb", "thompson", "moucb"]
TAGS = ["code", "explain", "tests"]
METHOD_COLORS = {
    "epsilon": "#ff7f0e",
    "ppo": "#1f77b4",
    "softmax": "#9467bd",
    "ucb": "#2ca02c",
    "thompson": "#b8860b",
    "moucb": "#d62728",
}


def safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def is_valid_number(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def mean(xs):
    vals = [x for x in xs if is_valid_number(x)]
    return float(sum(vals) / len(vals)) if vals else math.nan


def percentile(xs, q):
    vals = [x for x in xs if is_valid_number(x)]
    return float(np.percentile(vals, q)) if vals else math.nan


def bootstrap_mean_ci(values, n_boot=2000, ci=95, seed=42):
    vals = np.array([v for v in values if is_valid_number(v)], dtype=float)
    if len(vals) == 0:
        return math.nan, math.nan
    if len(vals) == 1:
        return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(seed)
    means = []
    n = len(vals)
    for _ in range(n_boot):
        sample = rng.choice(vals, size=n, replace=True)
        means.append(np.mean(sample))
    alpha = (100 - ci) / 2.0
    return float(np.percentile(means, alpha)), float(np.percentile(means, 100 - alpha))


def paired_cohens_d(xs, ys):
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    if len(xs) == 0 or len(xs) != len(ys):
        return math.nan
    diff = xs - ys
    if len(diff) < 2:
        return 0.0
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diff) / sd)


def rank_biserial_from_diffs(diffs):
    diffs = np.array([d for d in diffs if is_valid_number(d) and d != 0], dtype=float)
    if len(diffs) == 0:
        return 0.0
    pos = np.sum(diffs > 0)
    neg = np.sum(diffs < 0)
    return float((pos - neg) / (pos + neg))


def find_preferred_logs(logdir):
    files = glob.glob(os.path.join(logdir, "route_log*.jsonl"))
    specific = defaultdict(dict)
    generic = {}
    specific_re = re.compile(
        r"route_log_(epsilon|ppo|softmax|ucb|thompson|moucb)_(code|explain|tests)_.*\.jsonl$",
        re.I,
    )
    generic_re = re.compile(r"route_log_(epsilon|ppo|softmax|ucb|thompson|moucb)\.jsonl$", re.I)

    for f in files:
        name = os.path.basename(f)
        m = specific_re.match(name)
        if m:
            specific[m.group(1).lower()][m.group(2).lower()] = f
            continue
        m = generic_re.match(name)
        if m:
            generic[m.group(1).lower()] = f

    selected = []
    for method in ALL_METHODS:
        if specific.get(method):
            for tag in TAGS:
                if tag in specific[method]:
                    selected.append(specific[method][tag])
        elif method in generic:
            selected.append(generic[method])
    return selected


def parse_method_and_tag_from_name(path):
    name = os.path.basename(path).lower()
    method = "unknown"
    tag = None
    for m in ALL_METHODS:
        if f"route_log_{m}" in name:
            method = m
            break
    for t in TAGS:
        if f"_{t}_" in name:
            tag = t
            break
    return method, tag


def get_first_present(row, keys):
    for k in keys:
        if k in row and row.get(k) is not None:
            return row.get(k)
    return None


def infer_tag_from_row(row):
    raw = get_first_present(row, ["step_tag", "tag", "prompt_tag", "task_tag", "category"])
    if raw is None:
        return None
    raw = str(raw).strip().lower()
    if raw in TAGS:
        return raw
    return None


def load_records(files):
    records = []
    for f in files:
        method, filename_tag = parse_method_and_tag_from_name(f)
        row_counter = 0
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue

                tag = infer_tag_from_row(r) or filename_tag
                if tag not in TAGS:
                    continue

                lat = safe_float(get_first_present(r, ["latency_s", "latency", "exec_ms", "latency_ms"]))
                if is_valid_number(lat) and get_first_present(r, ["exec_ms", "latency_ms"]) is not None:
                    # Convert ms to seconds when ms fields are used.
                    lat = lat / 1000.0

                rew = safe_float(get_first_present(r, ["reward", "score"]))
                qual = safe_float(get_first_present(r, ["quality", "quality_score", "structural_quality"]))
                cost = safe_float(get_first_present(r, ["cost", "cost_estimate", "token_cost"]))
                backend = get_first_present(r, ["backend", "backend_id", "selected_backend"])

                step_id = get_first_present(r, ["step_id", "request_id", "task_id"])
                # Many logs do not share matching step_ids across methods.
                # We therefore also keep a sequential index within each file.
                sequential_idx = row_counter
                row_counter += 1

                records.append(
                    {
                        "method": method,
                        "tag": tag,
                        "step_id": step_id,
                        "seq_idx": sequential_idx,
                        "latency_s": lat,
                        "reward": rew,
                        "quality": qual,
                        "cost": cost,
                        "backend": backend,
                        "source_file": os.path.basename(f),
                    }
                )
    return records


def summarize_by_method_tag(records):
    rows = []
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["method"], r["tag"])].append(r)

    for method in ALL_METHODS:
        for tag in TAGS:
            grp = grouped.get((method, tag), [])
            rows.append(
                {
                    "method": method,
                    "tag": tag,
                    "n": len(grp),
                    "mean_latency_s": mean([r["latency_s"] for r in grp]),
                    "p50_latency_s": percentile([r["latency_s"] for r in grp], 50),
                    "p95_latency_s": percentile([r["latency_s"] for r in grp], 95),
                    "mean_reward": mean([r["reward"] for r in grp]),
                    "mean_quality": mean([r["quality"] for r in grp]),
                    "mean_cost": mean([r["cost"] for r in grp]),
                }
            )
    return rows


def build_pairs(records, metric):
    # Pair by order within each tag. This is more robust than step_id for these logs.
    by_method_tag = defaultdict(list)
    for r in records:
        val = r.get(metric)
        if not is_valid_number(val):
            continue
        by_method_tag[(r["method"], r["tag"])].append(val)

    pairs = defaultdict(lambda: defaultdict(lambda: {"ppo": [], "base": []}))
    for tag in TAGS:
        ppo_vals = by_method_tag.get(("ppo", tag), [])
        for baseline in BASELINES:
            base_vals = by_method_tag.get((baseline, tag), [])
            n = min(len(ppo_vals), len(base_vals))
            if n == 0:
                continue
            pairs[tag][baseline]["ppo"] = ppo_vals[:n]
            pairs[tag][baseline]["base"] = base_vals[:n]
    return pairs


def run_pairwise_stats(records):
    out_rows = []
    for metric in ["quality", "latency_s", "reward"]:
        pairs = build_pairs(records, metric)
        for tag in TAGS:
            for baseline in BASELINES:
                ppo_vals = pairs[tag][baseline]["ppo"]
                base_vals = pairs[tag][baseline]["base"]
                n = min(len(ppo_vals), len(base_vals))
                if n == 0:
                    out_rows.append(
                        {
                            "tag": tag,
                            "metric": metric,
                            "baseline": baseline,
                            "n_pairs": 0,
                            "ppo_mean": math.nan,
                            "baseline_mean": math.nan,
                            "mean_diff": math.nan,
                            "ci_low": math.nan,
                            "ci_high": math.nan,
                            "wilcoxon_stat": math.nan,
                            "p_value": math.nan,
                            "cohens_d_paired": math.nan,
                            "rank_biserial": math.nan,
                        }
                    )
                    continue

                ppo_arr = np.array(ppo_vals, dtype=float)
                base_arr = np.array(base_vals, dtype=float)
                diffs = ppo_arr - base_arr

                try:
                    # If all differences are zero, scipy may error.
                    if np.allclose(diffs, 0):
                        stat = 0.0
                        pval = 1.0
                    else:
                        w = wilcoxon(
                            ppo_arr,
                            base_arr,
                            zero_method="wilcox",
                            alternative="two-sided",
                            method="auto",
                        )
                        stat = float(w.statistic)
                        pval = float(w.pvalue)
                except Exception:
                    stat = math.nan
                    pval = math.nan

                ci_low, ci_high = bootstrap_mean_ci(diffs)
                out_rows.append(
                    {
                        "tag": tag,
                        "metric": metric,
                        "baseline": baseline,
                        "n_pairs": n,
                        "ppo_mean": mean(ppo_vals),
                        "baseline_mean": mean(base_vals),
                        "mean_diff": mean(diffs),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "wilcoxon_stat": stat,
                        "p_value": pval,
                        "cohens_d_paired": paired_cohens_d(ppo_arr, base_arr),
                        "rank_biserial": rank_biserial_from_diffs(diffs),
                    }
                )
    return out_rows


def save_csv(rows, path, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def fmt(x, digits=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{digits}f}"


def save_markdown(stats_rows, path):
    headers = [
        "Tag",
        "Metric",
        "Baseline",
        "Pairs",
        "PPO Mean",
        "Baseline Mean",
        "Mean Diff",
        "95% CI",
        "p-value",
        "Cohen's d",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join([":--"] * len(headers)) + "|\n")
        for r in stats_rows:
            f.write(
                "| "
                + " | ".join(
                    [
                        r["tag"],
                        r["metric"],
                        r["baseline"],
                        str(r["n_pairs"]),
                        fmt(r["ppo_mean"], 3),
                        fmt(r["baseline_mean"], 3),
                        fmt(r["mean_diff"], 3),
                        f"[{fmt(r['ci_low'],3)}, {fmt(r['ci_high'],3)}]",
                        fmt(r["p_value"], 4),
                        fmt(r["cohens_d_paired"], 3),
                    ]
                )
                + " |\n"
            )


def save_latex(stats_rows, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lllrllllll}\n")
        f.write("\\toprule\n")
        f.write(
            "Tag & Metric & Baseline & Pairs & PPO Mean & Baseline Mean & Mean Diff & 95\\% CI & p-value & Cohen's d \\\\\n"
        )
        f.write("\\midrule\n")
        for r in stats_rows:
            ci = f"[{fmt(r['ci_low'],3)}, {fmt(r['ci_high'],3)}]"
            f.write(
                f"{r['tag']} & {r['metric']} & {r['baseline']} & {r['n_pairs']} & "
                f"{fmt(r['ppo_mean'],3)} & {fmt(r['baseline_mean'],3)} & "
                f"{fmt(r['mean_diff'],3)} & {ci} & {fmt(r['p_value'],4)} & "
                f"{fmt(r['cohens_d_paired'],3)} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def build_grouped_metric_values(records):
    d = defaultdict(list)
    for r in records:
        for metric in ["quality", "latency_s", "reward"]:
            val = r.get(metric)
            if not is_valid_number(val):
                continue
            d[(r["tag"], r["method"], metric)].append(val)
    return d


def quality_ci_plot(summary_rows, grouped_metric_values, outpath):
    fig, axes = plt.subplots(1, len(TAGS), figsize=(15, 4.8), sharey=True)
    if len(TAGS) == 1:
        axes = [axes]

    for ax, tag in zip(axes, TAGS):
        tag_rows = [r for r in summary_rows if r["tag"] == tag]
        methods = [r["method"] for r in tag_rows]
        means = [r["mean_quality"] for r in tag_rows]
        lowers = []
        uppers = []

        for m in methods:
            vals = grouped_metric_values[(tag, m, "quality")]
            lo, hi = bootstrap_mean_ci(vals)
            mu = mean(vals)
            if not is_valid_number(mu) or not is_valid_number(lo) or not is_valid_number(hi):
                lowers.append(0.0)
                uppers.append(0.0)
            else:
                lowers.append(max(0.0, mu - lo))
                uppers.append(max(0.0, hi - mu))

        x = np.arange(len(methods))
        ax.bar(
            x,
            means,
            color=[METHOD_COLORS[m] for m in methods],
            edgecolor="black",
            linewidth=0.6,
        )
        ax.errorbar(
            x,
            means,
            yerr=[lowers, uppers],
            fmt="none",
            ecolor="black",
            capsize=4,
            linewidth=1.0,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.set_title(tag)
        ax.set_ylabel("Mean Quality")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def effect_size_plot(stats_rows, metric, outpath):
    rows = [r for r in stats_rows if r["metric"] == metric]
    fig, axes = plt.subplots(1, len(TAGS), figsize=(15, 4.8), sharey=True)
    if len(TAGS) == 1:
        axes = [axes]

    for ax, tag in zip(axes, TAGS):
        subset = [r for r in rows if r["tag"] == tag]
        baselines = [r["baseline"] for r in subset]
        vals = [r["cohens_d_paired"] for r in subset]
        x = np.arange(len(baselines))
        ax.bar(
            x,
            vals,
            color=[METHOD_COLORS[b] for b in baselines],
            edgecolor="black",
            linewidth=0.6,
        )
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(baselines, rotation=30, ha="right")
        ax.set_title(tag)
        ax.set_ylabel("Paired Cohen's d")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def print_debug_summary(records):
    print(f"Loaded records: {len(records)}")
    by_method_tag = defaultdict(int)
    for r in records:
        by_method_tag[(r["method"], r["tag"])] += 1

    print("Counts by method/tag:")
    for method in ALL_METHODS:
        for tag in TAGS:
            print(f"  {method:9s} {tag:8s}: {by_method_tag[(method, tag)]}")

    print("Mean quality by method/tag:")
    grouped = defaultdict(list)
    for r in records:
        if is_valid_number(r.get("quality")):
            grouped[(r["method"], r["tag"])].append(r["quality"])

    for method in ALL_METHODS:
        for tag in TAGS:
            vals = grouped.get((method, tag), [])
            print(f"  {method:9s} {tag:8s}: {fmt(mean(vals), 3)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="/mnt/data")
    ap.add_argument("--outdir", default="/mnt/data/stats_results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    files = find_preferred_logs(args.logdir)

    print("Using files:")
    for f in files:
        print(" -", os.path.basename(f))

    records = load_records(files)
    if not records:
        raise SystemExit("No records loaded. Check filenames and field names.")

    grouped_metric_values = build_grouped_metric_values(records)
    print_debug_summary(records)

    summary_rows = summarize_by_method_tag(records)
    stats_rows = run_pairwise_stats(records)

    save_csv(summary_rows, os.path.join(args.outdir, "per_task_summary.csv"))
    save_csv(stats_rows, os.path.join(args.outdir, "ppo_pairwise_stats.csv"))
    save_markdown(stats_rows, os.path.join(args.outdir, "ppo_pairwise_stats.md"))
    save_latex(stats_rows, os.path.join(args.outdir, "ppo_pairwise_stats.tex"))

    quality_ci_plot(
        summary_rows,
        grouped_metric_values,
        os.path.join(args.outdir, "quality_ci_by_tag.pdf"),
    )
    effect_size_plot(
        stats_rows,
        "quality",
        os.path.join(args.outdir, "effect_sizes_quality.pdf"),
    )
    effect_size_plot(
        stats_rows,
        "latency_s",
        os.path.join(args.outdir, "effect_sizes_latency.pdf"),
    )
    effect_size_plot(
        stats_rows,
        "reward",
        os.path.join(args.outdir, "effect_sizes_reward.pdf"),
    )

    print(f"Wrote outputs to {args.outdir}")


if __name__ == "__main__":
    main()
