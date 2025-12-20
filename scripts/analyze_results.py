#!/usr/bin/env python3
import argparse, glob, json, os, sys, statistics as st, math, csv
from collections import defaultdict
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# ====== Global plotting style ======
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.labelweight": "bold",
    "axes.titlepad": 0,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

# ====== Method lists / palette & styles ======
ALL_METHODS = ["epsilon", "ppo", "softmax", "ucb", "thompson"]

# Bold, consistent colors for all plots
METHOD_COLORS = {
    "epsilon":   "#ff7f0e",  # bold orange
    "ppo":       "#1f77b4",  # bold blue
    "softmax":   "#9467bd",  # bold purple
    "ucb":       "#2ca02c",  # bold green
    "thompson":  "#b8860b",  # dark mustard
}

# Bar hatching so B/W prints are still readable
METHOD_HATCHES = {
    "epsilon":   "//",
    "ppo":       "\\\\",
    "softmax":   "xx",
    "ucb":       "-",   # changed from dots to horizontal lines
    "thompson":  "++",
}

METHOD_MARKERS = {
    "epsilon":   "s",   # square
    "ppo":       "o",   # dot / circle
    "softmax":   "^",   # triangle
    "ucb":       "D",   # diamond
    "thompson":  "X",   # X marker
}

GRID_KW = dict(linestyle="--", alpha=0.35, linewidth=0.8)
BAR_KW  = dict(edgecolor="black", linewidth=0.6)

# ====== IO helpers ======
def find_logs(auto_dir="logs"):
    """
    Find all router logs for known methods.
    patterns:
      logs/route_*_epsilon_*.jsonl
      logs/route_*_ppo_*.jsonl
      logs/route_*_softmax_*.jsonl
      logs/route_*_ucb_*.jsonl
      logs/route_*_thompson_*.jsonl
    """
    methods = ALL_METHODS
    patterns = [
        os.path.join(auto_dir, f"route_*_{m}_*.jsonl") for m in methods
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    return sorted(set(files))

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    continue

def method_from_name(path):
    name = os.path.basename(path).lower()
    if "epsilon" in name:   return "epsilon"
    if "softmax" in name:   return "softmax"
    if "ucb" in name:       return "ucb"
    if "thompson" in name:  return "thompson"
    if "ppo" in name:       return "ppo"
    return "unknown"

# ====== aggregation ======
def collect_records(files):
    recs = []
    for f in files:
        method = method_from_name(f)
        for r in load_jsonl(f):
            r["__method__"] = method
            if "backend" in r:
                r["backend"] = str(r.get("backend"))
            r["step_tag"]   = (r.get("step_tag") or "unknown")
            r["latency_s"]  = safe_float(r.get("latency_s"))
            r["reward"]     = safe_float(r.get("reward"))
            r["quality"]    = safe_float(r.get("quality"))
            r["cost"]       = safe_float(r.get("cost"))
            recs.append(r)
    return recs

def pct95(vals):
    if not vals: return math.nan
    s = sorted(vals)
    return s[int(0.95*(len(s)-1))]

def pct50(vals):
    if not vals: return math.nan
    s = sorted(vals)
    return s[int(0.50*(len(s)-1))]

def summarize(records):
    vals = lambda key: [safe_float(r.get(key)) for r in records if safe_float(r.get(key)) is not None]
    lat  = vals("latency_s")
    rew  = vals("reward")
    qual = vals("quality")
    cost = vals("cost")
    backs= [r["backend"] for r in records if r.get("backend") is not None]

    s = {
        "n": len(records),
        "mean_latency": st.mean(lat) if lat else math.nan,
        "p50_latency": pct50(lat),
        "p95_latency": pct95(lat),
        "mean_reward": st.mean(rew) if rew else math.nan,
        "mean_quality": st.mean(qual) if qual else math.nan,
        "mean_cost": st.mean(cost) if cost else math.nan,
        "backend_counts": {b: backs.count(b) for b in sorted(set(backs))},
    }
    return s

def group_by(items, keys):
    d = defaultdict(list)
    for r in items:
        k = tuple(r.get(k) for k in keys)
        d[k].append(r)
    return d

# ====== CSV ======
def save_csv(summary_map, out_csv):
    fields = ["method","n","mean_latency","p50_latency","p95_latency","mean_reward","mean_quality","mean_cost","backend_counts"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for method, s in summary_map.items():
            row = {"method": method}
            row.update(s)
            w.writerow(row)

# ====== color helper using unified palette ======
def color_for(method):
    return METHOD_COLORS.get(method, "#bbbbbb")

# ====== plots ======
def overall_backend_usage(summary_map, outpath):
    methods = [m for m in ALL_METHODS if m in summary_map]
    if not methods:
        return

    backends = sorted(set().union(*[summary_map[m]["backend_counts"].keys() for m in methods]))
    x = np.arange(len(backends))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, m in enumerate(methods):
        counts = [summary_map[m]["backend_counts"].get(b, 0) for b in backends]
        bars = ax.bar(
            x + (i - (len(methods) - 1) / 2) * width,
            counts,
            width,
            label=m,
            color=color_for(m),
            **BAR_KW,
        )
        hatch = METHOD_HATCHES.get(m)
        if hatch:
            for b in bars:
                b.set_hatch(hatch)

    ax.set_xticks(x)
    ax.set_xticklabels(backends)
    ax.set_xlabel("Backend ID", fontweight="bold")
    ax.set_ylabel("Number of Requests", fontweight="bold")

    ax.grid(True, axis="y", **GRID_KW)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def per_request_pareto_all(records, outpath):
    """
    Per-request Pareto cloud for ALL methods, using METHOD_COLORS
    and distinct marker shapes. PPO gets dots, others get different shapes.
    """
    methods = sorted(set(r.get("__method__") for r in records))
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    for m in methods:
        xs, ys = [], []
        for r in records:
            if r.get("__method__") != m:
                continue
            lat = safe_float(r.get("latency_s"))
            rew = safe_float(r.get("reward"))
            if lat is None or rew is None:
                continue
            xs.append(lat)
            ys.append(rew)
        if not xs:
            continue

        marker = METHOD_MARKERS.get(m, "o")  # default to dot if unknown

        ax.scatter(
            xs,
            ys,
            s=16 if m == "ppo" else 14,   # PPO dots maybe slightly larger
            alpha = 0.85 if m != "ppo" else 0.75,
            label=m,
            color=color_for(m),
            marker=marker,
            edgecolors="none",
        )

    ax.set_xlabel("Latency (s) ↓", fontweight="bold")
    ax.set_ylabel("Reward ↑", fontweight="bold")
    ax.grid(True, **GRID_KW)
    ax.legend(title="Method", frameon=False)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def reward_distribution(records, outpath):
    """
    Reward distribution per method:
    boxplot + jittered points.
    - Box fill: method color (bold)
    - Jitter dots: black so they stand out
    - Median line: black
    """
    methods = sorted(set(r.get("__method__") for r in records))
    data = {m: [] for m in methods}
    for r in records:
        m = r.get("__method__")
        rew = safe_float(r.get("reward"))
        if m is None or rew is None:
            continue
        data[m].append(rew)

    methods_present = [m for m in methods if data[m]]
    if not methods_present:
        return

    pos = np.arange(len(methods_present))
    fig, ax = plt.subplots(figsize=(7.5, 5))

    # Boxplot
    bp = ax.boxplot(
        [data[m] for m in methods_present],
        positions=pos,
        widths=0.55,
        showfliers=False,
        patch_artist=True,
    )

    # Color boxes and medians
    for box, m in zip(bp["boxes"], methods_present):
        box.set_facecolor(color_for(m))
        box.set_alpha(0.85)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2.0)

    # Jittered points (black dots to stand out)
    for i, m in enumerate(methods_present):
        ys = data[m]
        xs = np.random.normal(loc=pos[i], scale=0.05, size=len(ys))
        ax.scatter(
            xs,
            ys,
            s=7,
            alpha=0.6,
            color="black",
            edgecolors="none",
        )

    ax.set_xticks(pos)
    ax.set_xticklabels(methods_present)
    ax.set_ylabel("Reward ↑", fontweight="bold")
    #ax.set_title("Reward Distribution per Method")
    ax.grid(True, axis="y", **GRID_KW)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def latency_distribution(records, outpath):
    """
    Latency distribution per method:
    boxplot + jittered points.
    - Box fill: method color (bold)
    - Jitter dots: black
    - Median line: black
    """
    methods = sorted(set(r.get("__method__") for r in records))
    data = {m: [] for m in methods}
    for r in records:
        m = r.get("__method__")
        lat = safe_float(r.get("latency_s"))
        if m is None or lat is None:
            continue
        data[m].append(lat)

    methods_present = [m for m in methods if data[m]]
    if not methods_present:
        return

    pos = np.arange(len(methods_present))
    fig, ax = plt.subplots(figsize=(7.5, 5))

    bp = ax.boxplot(
        [data[m] for m in methods_present],
        positions=pos,
        widths=0.55,
        showfliers=False,
        patch_artist=True,
    )

    for box, m in zip(bp["boxes"], methods_present):
        box.set_facecolor(color_for(m))
        box.set_alpha(0.85)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2.0)

    for i, m in enumerate(methods_present):
        ys = data[m]
        xs = np.random.normal(loc=pos[i], scale=0.05, size=len(ys))
        ax.scatter(
            xs,
            ys,
            s=7,
            alpha=0.6,
            color="black",
            edgecolors="none",
        )

    ax.set_xticks(pos)
    ax.set_xticklabels(methods_present)
    ax.set_ylabel("Latency (s) ↓", fontweight="bold")
    # ax.set_title("Latency Distribution per Method")
    ax.grid(True, axis="y", **GRID_KW)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def bytag_bar_line(by_method_tag, outpath):
    """
    Show ONLY mean latency per method, per tag (code/explain/tests),
    with all methods included.
    """
    tags = sorted({k[1] for k in by_method_tag.keys()})
    if not tags:
        return

    fig, axes = plt.subplots(
        1, len(tags),
        figsize=(6 * len(tags), 4.8),
        sharey=False
    )
    if len(tags) == 1:
        axes = [axes]

    for ax, tag in zip(axes, tags):
        methods_present = [m for m in ALL_METHODS if (m, tag) in by_method_tag]
        if not methods_present:
            ax.axis("off")
            continue

        xs = []
        lat_vals = []

        for m in methods_present:
            grp = by_method_tag.get((m, tag))
            if not grp:
                continue
            s = summarize(grp)
            xs.append(m.upper())
            lat_vals.append(s["mean_latency"])

        bars = ax.bar(
            xs,
            lat_vals,
            color=[color_for(m) for m in methods_present],
            **BAR_KW,
        )

        for bar, m in zip(bars, methods_present):
            hatch = METHOD_HATCHES.get(m)
            if hatch:
                bar.set_hatch(hatch)

        ax.set_xlabel(f"Method ({tag})", fontweight="bold")
        ax.set_ylabel("Mean Latency (s)", fontweight="bold")
        # ax.set_title(f"Mean Latency by Method ({tag})", fontweight="bold")

        ax.grid(True, axis="y", **GRID_KW)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def bytag_backend_usage(by_method_tag, outpath):
    """
    Backend usage by tag with ALL methods, bar+hatch so B/W is readable.
    """
    tags = sorted({k[1] for k in by_method_tag.keys()})
    if not tags:
        return

    fig, axes = plt.subplots(
        1, len(tags),
        figsize=(6 * len(tags), 4.8),
        sharey=False
    )
    if len(tags) == 1:
        axes = [axes]

    for ax, tag in zip(axes, tags):
        methods = [m for m in ALL_METHODS if (m, tag) in by_method_tag]
        if not methods:
            ax.axis("off")
            continue

        backs_union = set()
        for m in methods:
            grp = by_method_tag.get((m, tag))
            if not grp:
                continue
            s = summarize(grp)
            backs_union |= set(s["backend_counts"].keys())

        backs = sorted(backs_union)
        if not backs:
            ax.axis("off")
            continue

        x = np.arange(len(backs))
        width = 0.8 / max(len(methods), 1)

        for i, m in enumerate(methods):
            grp = by_method_tag.get((m, tag))
            if grp:
                s = summarize(grp)
                counts = [s["backend_counts"].get(b, 0) for b in backs]
            else:
                counts = [0] * len(backs)

            bars = ax.bar(
                x + (i - (len(methods) - 1) / 2) * width,
                counts,
                width,
                color=color_for(m),
                label=m,
                **BAR_KW,
            )

            hatch = METHOD_HATCHES.get(m)
            if hatch:
                for b in bars:
                    b.set_hatch(hatch)

        ax.set_xticks(x)
        ax.set_xticklabels(backs)
        ax.set_xlabel(f"Backend ({tag})", fontweight="bold")
        ax.set_ylabel("Count", fontweight="bold")

        ax.grid(True, axis="y", **GRID_KW)
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close()

# ====== comparison table (Markdown + LaTeX) ======
def fmt(x, d=3):
    return f"{x:.{d}f}" if x is not None and not math.isnan(x) else "–"

def make_pretty_table(overall, out_md, out_tex):
    methods = [m for m in ALL_METHODS if m in overall]
    if not methods:
        print("⚠️ No known methods found for table.")
        return

    # Markdown
    header = (
        "# Overall Routing Performance by Method\n\n"
        "| Method | n | Mean Latency (s) | p50 (s) | p95 (s) | Mean Reward | "
        "Mean Quality | Mean Cost |\n"
        "|:--|--:|--:|--:|--:|--:|--:|--:|\n"
    )
    rows = []
    for m in methods:
        s = overall[m]
        rows.append(
            f"| {m} | {s['n']} | {fmt(s['mean_latency'])} | "
            f"{fmt(s['p50_latency'])} | {fmt(s['p95_latency'])} | "
            f"{fmt(s['mean_reward'])} | {fmt(s['mean_quality'])} | "
            f"{fmt(s['mean_cost'])} |"
        )
    md = header + "\n".join(rows) + "\n"

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)

    # LaTeX
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Method & $n$ & Mean Lat. & p50 & p95 & Mean Rew. & Mean Qual. & Mean Cost \\",
        r"\midrule",
    ]

    for m in methods:
        s = overall[m]
        tex_lines.append(
            f"{m} & {s['n']} & "
            f"{fmt(s['mean_latency'])} & "
            f"{fmt(s['p50_latency'])} & "
            f"{fmt(s['p95_latency'])} & "
            f"{fmt(s['mean_reward'])} & "
            f"{fmt(s['mean_quality'])} & "
            f"{fmt(s['mean_cost'])} \\\\"
        )

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Overall routing performance across all methods. "
        r"Lower latency and cost, and higher reward and quality are desirable.}",
        r"\end{table}",
        ""
    ])

    tex = "\n".join(tex_lines)
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(tex)

    print(f"📊 Saved summary tables:\n  {out_md}\n  {out_tex}")
    print("\n" + md)

# ====== headline winners (PPO vs epsilon, console only) ======
def winner_name(metric_ppo, metric_eps, higher_is_better):
    if any(v is None or math.isnan(v) for v in (metric_ppo, metric_eps)):
        return "tie"
    if higher_is_better:
        return "ppo" if metric_ppo > metric_eps else ("epsilon" if metric_eps > metric_ppo else "tie")
    else:
        return "ppo" if metric_ppo < metric_eps else ("epsilon" if metric_eps < metric_ppo else "tie")

# ====== main ======
def main():
    ap = argparse.ArgumentParser(description="Analyze router JSONL logs.")
    ap.add_argument("--outdir", default="results", help="Where to write CSV/PDF outputs.")
    args = ap.parse_args()

    files = find_logs()
    if not files:
        print("No log files found under ./logs/")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    records = collect_records(files)
    if not records:
        print("No records parsed.")
        sys.exit(2)

    by_method = group_by(records, ["__method__"])
    overall = {k[0]: summarize(v) for k, v in by_method.items()}

    by_method_tag = group_by(records, ["__method__", "step_tag"])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.outdir, f"summaries_{ts}.csv")
    save_csv(overall, csv_path)

    # Plots (note: overall_bar_metrics removed)
    overall_backend_usage(overall, os.path.join(args.outdir, "overall_backend_usage.pdf"))
    per_request_pareto_all(records, os.path.join(args.outdir, "per_request_pareto_all_methods.pdf"))
    reward_distribution(records, os.path.join(args.outdir, "reward_distribution_by_method.pdf"))
    latency_distribution(records, os.path.join(args.outdir, "latency_distribution_by_method.pdf"))
    bytag_bar_line(by_method_tag, os.path.join(args.outdir, "bytag_latency_reward.pdf"))
    bytag_backend_usage(by_method_tag, os.path.join(args.outdir, "bytag_backend_usage.pdf"))

    make_pretty_table(
        overall,
        out_md=os.path.join(args.outdir, "comparison_table.md"),
        out_tex=os.path.join(args.outdir, "comparison_table.tex"),
    )

    print(f"✅ Wrote {csv_path}")
    print(f"✅ Wrote PDFs to {args.outdir}/:")
    print("   - overall_backend_usage.pdf")
    print("   - overall_pareto_reward_vs_latency.pdf")
    print("   - per_request_pareto_all_methods.pdf")
    print("   - reward_distribution_by_method.pdf")
    print("   - latency_distribution_by_method.pdf")
    print("   - bytag_latency_reward.pdf")
    print("   - bytag_backend_usage.pdf")
    print("   - comparison_table.md / comparison_table.tex")

    print("\n=== Summary ===")
    for m in sorted(overall.keys()):
        s = overall[m]
        print(f"[{m}] n={s['n']}")
        print(f"  mean latency: {s['mean_latency']:.3f}s   p50: {s['p50_latency']:.3f}s   p95: {s['p95_latency']:.3f}s")
        print(f"  mean reward : {s['mean_reward']:.3f}    mean quality: {s['mean_quality']:.3f}")
        print(f"  mean cost   : {s['mean_cost']:.3f}")
        print(f"  backend counts: {s['backend_counts']}")

    if "ppo" in overall and "epsilon" in overall:
        lat_winner = winner_name(overall["ppo"]["mean_latency"], overall["epsilon"]["mean_latency"], higher_is_better=False)
        rew_winner = winner_name(overall["ppo"]["mean_reward"],  overall["epsilon"]["mean_reward"],  higher_is_better=True)
        qual_winner= winner_name(overall["ppo"]["mean_quality"], overall["epsilon"]["mean_quality"], higher_is_better=True)
        cost_winner= winner_name(overall["ppo"]["mean_cost"],    overall["epsilon"]["mean_cost"],    higher_is_better=False)
        print("\n=== Headline (PPO vs epsilon) ===")
        print(f"Latency winner : {lat_winner} (lower is better)")
        print(f"Reward winner  : {rew_winner} (higher is better)")
        print(f"Quality winner : {qual_winner} (higher is better)")
        print(f"Cost winner    : {cost_winner} (lower is better)")

if __name__ == "__main__":
    main()
