#!/usr/bin/env python3
import argparse, glob, json, os, sys, statistics as st, math, csv
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ====== Global plotting style (bigger, bold fonts, no in-figure titles) ======
plt.rcParams.update({
    "font.size": 16,              # base font size
    "axes.labelsize": 16,
    "axes.labelweight": "bold",
    "axes.titlepad": 0,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

# ====== Fixed palette & styles ======
PPO_COLOR     = "#1f77b4"  # blue
EPS_COLOR     = "#ff7f0e"  # orange
LINE_COLOR    = "#3b3b3b"
GRID_KW       = dict(linestyle="--", alpha=0.35, linewidth=0.8)
BAR_KW        = dict(edgecolor="black", linewidth=0.6)

# ====== IO helpers ======
def find_logs(auto_dir="logs"):
    patterns = [
        os.path.join(auto_dir, "route_*_ppo_*.jsonl"),
        os.path.join(auto_dir, "route_*_epsilon_*.jsonl"),
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
    if "epsilon" in name: return "epsilon"
    if "ppo" in name:     return "ppo"
    return "unknown"

# ====== aggregation ======
def collect_records(files):
    recs = []
    for f in files:
        method = method_from_name(f)
        for r in load_jsonl(f):
            r["__method__"] = method
            # normalize types
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
    """Group records by a list of keys (e.g., ['__method__', 'step_tag'])."""
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

# ====== plots ======
def color_for(method):
    return PPO_COLOR if method == "ppo" else EPS_COLOR

def overall_bar_line(summary_map, outpath):
    methods = [m for m in ["epsilon","ppo"] if m in summary_map]
    if not methods:
        return

    lat = [summary_map[m]["mean_latency"] for m in methods]
    rew = [summary_map[m]["mean_reward"] for m in methods]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    bars = ax1.bar(methods, lat, color=[color_for(m) for m in methods], **BAR_KW)
    ax2.plot(methods, rew, marker="o", markersize=8, linewidth=2.5, color=LINE_COLOR)

    ax1.set_xlabel("Routing Method", fontweight="bold")
    ax1.set_ylabel("Mean Latency (s)", fontweight="bold")
    ax2.set_ylabel("Mean Reward", fontweight="bold")

    ax1.grid(True, axis="y", **GRID_KW)
    ax1.tick_params(axis="both", labelsize=12)
    ax2.tick_params(axis="y", labelsize=12)

    for b, v in zip(bars, lat):
        ax1.text(b.get_x() + b.get_width() / 2,
                 v * 1.01,
                 f"{v:.2f}",
                 fontsize=11, fontweight="bold",
                 ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def overall_backend_usage(summary_map, outpath):
    methods = [m for m in ["epsilon","ppo"] if m in summary_map]
    if not methods:
        return

    backends = sorted(set().union(*[summary_map[m]["backend_counts"].keys() for m in methods]))
    x = np.arange(len(backends))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, m in enumerate(methods):
        counts = [summary_map[m]["backend_counts"].get(b, 0) for b in backends]
        ax.bar(x + (i - 0.5) * width, counts, width,
               label=m, color=color_for(m), **BAR_KW)

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

def overall_pareto(summary_map, outpath):
    methods = [m for m in ["epsilon","ppo"] if m in summary_map]
    if not methods:
        return

    fig, ax = plt.subplots(figsize=(7.5, 5))

    for m in methods:
        x = summary_map[m]["mean_latency"]
        y = summary_map[m]["mean_reward"]

        ax.scatter([x], [y], s=140,
                   color=color_for(m),
                   edgecolor="black",
                   linewidth=0.9,
                   zorder=3)

        ax.annotate(m.upper(), (x, y),
                    textcoords="offset points",
                    xytext=(8, 8),
                    fontsize=14,
                    fontweight="bold")

    ax.set_xlabel("Mean Latency (s) ↓", fontweight="bold")
    ax.set_ylabel("Mean Reward ↑", fontweight="bold")

    ax.grid(True, **GRID_KW)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def per_request_pareto(records, outpath):
    """
    Per-request Pareto cloud:
    one point per logged request, colored by method.
    Useful for deeper analysis or appendix figures.
    """
    methods = ["epsilon", "ppo"]
    fig, ax = plt.subplots(figsize=(6.8, 4.8))

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

        ax.scatter(
            xs,
            ys,
            s=10,
            alpha=0.4,
            color=color_for(m),
            label=m,
            edgecolors="none",
        )

    ax.set_xlabel("Latency (s) ↓")
    ax.set_ylabel("Reward ↑")
    ax.set_title("Per-request Reward vs Latency (Pareto cloud)")
    ax.grid(True, **GRID_KW)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def bytag_bar_line(by_method_tag, outpath):
    tags = sorted({k[1] for k in by_method_tag.keys()})
    methods = ["epsilon", "ppo"]
    if not tags:
        return

    fig, axes = plt.subplots(1, len(tags),
                             figsize=(6 * len(tags), 4.8),
                             sharey=False)
    if len(tags) == 1:
        axes = [axes]

    for ax, tag in zip(axes, tags):
        xs, lat_vals, rew_vals, colors = [], [], [], []

        for m in methods:
            grp = by_method_tag.get((m, tag))
            if not grp:
                continue
            s = summarize(grp)
            xs.append(m.upper())
            lat_vals.append(s["mean_latency"])
            rew_vals.append(s["mean_reward"])
            colors.append(color_for(m))

        if not xs:
            ax.axis("off")
            continue

        ax2 = ax.twinx()
        bars = ax.bar(xs, lat_vals, color=colors, **BAR_KW)
        ax2.plot(xs, rew_vals, marker="o", markersize=7,
                 linewidth=2.2, color=LINE_COLOR)

        ax.set_xlabel(f"Method ({tag})", fontweight="bold")
        ax.set_ylabel("Mean Latency (s)", fontweight="bold")
        ax2.set_ylabel("Mean Reward", fontweight="bold")

        ax.grid(True, axis="y", **GRID_KW)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax2.tick_params(axis="y", labelsize=12)

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def bytag_backend_usage(by_method_tag, outpath):
    tags = sorted({k[1] for k in by_method_tag.keys()})
    methods = ["epsilon", "ppo"]
    if not tags:
        return

    fig, axes = plt.subplots(1, len(tags),
                             figsize=(6 * len(tags), 4.8),
                             sharey=False)
    if len(tags) == 1:
        axes = [axes]

    for ax, tag in zip(axes, tags):
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
        width = 0.36

        for i, m in enumerate(methods):
            grp = by_method_tag.get((m, tag))
            if grp:
                s = summarize(grp)
                counts = [s["backend_counts"].get(b, 0) for b in backs]
            else:
                counts = [0] * len(backs)

            ax.bar(x + (i - 0.5) * width,
                   counts, width,
                   color=color_for(m),
                   **BAR_KW)

        ax.set_xticks(x)
        ax.set_xticklabels(backs)
        ax.set_xlabel(f"Backend ({tag})", fontweight="bold")
        ax.set_ylabel("Count", fontweight="bold")

        ax.grid(True, axis="y", **GRID_KW)
        ax.tick_params(axis="both", labelsize=12)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close()

# ====== comparison table (Markdown + LaTeX) ======
def fmt(x, d=3):
    return f"{x:.{d}f}" if x is not None and not math.isnan(x) else "–"

def pct_delta(new, base, invert=False):
    """Percent change vs base. If invert=True, 'lower is better' so flip sign."""
    if base in (None, 0) or math.isnan(base) or new is None or math.isnan(new):
        return None
    raw = (new - base) / abs(base) * 100.0
    return -raw if invert else raw

def make_pretty_table(overall, out_md, out_tex):
    if "ppo" not in overall or "epsilon" not in overall:
        print("⚠️ Need both PPO and epsilon summaries for table.")
        return

    eps = overall["epsilon"]
    ppo = overall["ppo"]

    d_lat = pct_delta(ppo["mean_latency"], eps["mean_latency"], invert=True)   # lower better
    d_p95 = pct_delta(ppo["p95_latency"],  eps["p95_latency"],  invert=True)   # lower better
    d_rew = pct_delta(ppo["mean_reward"],  eps["mean_reward"],  invert=False)  # higher better
    d_qual= pct_delta(ppo["mean_quality"], eps["mean_quality"], invert=False)  # higher better
    d_cost= pct_delta(ppo["mean_cost"],    eps["mean_cost"],    invert=True)   # lower better

    def winner(ppo_v, eps_v, higher_is_better):
        if any(v is None or math.isnan(v) for v in (ppo_v, eps_v)):
            return "tie"
        if higher_is_better:
            return "✅ PPO" if ppo_v > eps_v else ("⚠️ Epsilon" if eps_v > ppo_v else "tie")
        else:
            return "✅ PPO" if ppo_v < eps_v else ("⚠️ Epsilon" if eps_v < ppo_v else "tie")

    # ---------- Markdown table ----------
    md = f"""
# PPO vs Epsilon Comparative Metrics

| Metric | Epsilon | PPO | Direction | Δ% (vs Eps) | Winner |
|:--|--:|--:|:--:|--:|:--:|
| Mean Latency (s) | {fmt(eps['mean_latency'])} | {fmt(ppo['mean_latency'])} | ↓ | {('+' if d_lat is not None and d_lat>=0 else '') + (f'{d_lat:.1f}%' if d_lat is not None else '–')} | {winner(ppo['mean_latency'], eps['mean_latency'], False)} |
| p95 Latency (s)  | {fmt(eps['p95_latency'])}  | {fmt(ppo['p95_latency'])}  | ↓ | {('+' if d_p95 is not None and d_p95>=0 else '') + (f'{d_p95:.1f}%' if d_p95 is not None else '–')} | {winner(ppo['p95_latency'], eps['p95_latency'], False)} |
| Mean Reward      | {fmt(eps['mean_reward'])}  | {fmt(ppo['mean_reward'])}  | ↑ | {('+' if d_rew is not None and d_rew>=0 else '') + (f'{d_rew:.1f}%' if d_rew is not None else '–')} | {winner(ppo['mean_reward'], eps['mean_reward'], True)} |
| Mean Quality     | {fmt(eps['mean_quality'])} | {fmt(ppo['mean_quality'])} | ↑ | {('+' if d_qual is not None and d_qual>=0 else '') + (f'{d_qual:.1f}%' if d_qual is not None else '–')} | {winner(ppo['mean_quality'], eps['mean_quality'], True)} |
| Mean Cost        | {fmt(eps['mean_cost'])}    | {fmt(ppo['mean_cost'])}    | ↓ | {('+' if d_cost is not None and d_cost>=0 else '') + (f'{d_cost:.1f}%' if d_cost is not None else '–')} | {winner(ppo['mean_cost'], eps['mean_cost'], False)} |
| n (samples)      | {eps['n']} | {ppo['n']} | – | – | – |
""".strip() + "\n"

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)

    # ---------- LaTeX table ----------
    # Note: all literal % signs in the template use '%%' so Python's '%' formatter does not treat them as specifiers.
    tex = r"""
\begin{table}[t]
\centering
\small
\begin{tabular}{lrrrrc}
\toprule
Metric & Epsilon & PPO & Direction & $\Delta\%%$ (vs Eps) & Winner \\
\midrule
Mean Latency (s) & %s & %s & $\downarrow$ & %s & %s \\
p95 Latency (s)  & %s & %s & $\downarrow$ & %s & %s \\
Mean Reward      & %s & %s & $\uparrow$   & %s & %s \\
Mean Quality     & %s & %s & $\uparrow$   & %s & %s \\
Mean Cost        & %s & %s & $\downarrow$ & %s & %s \\
\midrule
n (samples)      & %d & %d & -- & -- & -- \\
\bottomrule
\end{tabular}
\caption{Comparison of PPO vs.\ $\varepsilon$-greedy across key metrics. Arrows indicate desired direction.}
\end{table}
""" % (
        fmt(eps['mean_latency']), fmt(ppo['mean_latency']),
        ('+' if d_lat is not None and d_lat>=0 else '') + (f'{d_lat:.1f}\\%%' if d_lat is not None else '--'),
        winner(ppo['mean_latency'], eps['mean_latency'], False),

        fmt(eps['p95_latency']), fmt(ppo['p95_latency']),
        ('+' if d_p95 is not None and d_p95>=0 else '') + (f'{d_p95:.1f}\\%%' if d_p95 is not None else '--'),
        winner(ppo['p95_latency'], eps['p95_latency'], False),

        fmt(eps['mean_reward']), fmt(ppo['mean_reward']),
        ('+' if d_rew is not None and d_rew>=0 else '') + (f'{d_rew:.1f}\\%%' if d_rew is not None else '--'),
        winner(ppo['mean_reward'], eps['mean_reward'], True),

        fmt(eps['mean_quality']), fmt(ppo['mean_quality']),
        ('+' if d_qual is not None and d_qual>=0 else '') + (f'{d_qual:.1f}\\%%' if d_qual is not None else '--'),
        winner(ppo['mean_quality'], eps['mean_quality'], True),

        fmt(eps['mean_cost']), fmt(ppo['mean_cost']),
        ('+' if d_cost is not None and d_cost>=0 else '') + (f'{d_cost:.1f}\\%%' if d_cost is not None else '--'),
        winner(ppo['mean_cost'], eps['mean_cost'], False),

        eps['n'], ppo['n']
    )

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(tex)

    print(f"📊 Saved summary tables:\n  {out_md}\n  {out_tex}")
    print("\n" + md)

# ====== headline winners ======
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

    # overall summaries by method
    by_method = group_by(records, ["__method__"])
    overall = {k[0]: summarize(v) for k, v in by_method.items()}

    # per-tag summaries (code/explain/tests) by method
    by_method_tag = group_by(records, ["__method__", "step_tag"])

    # Save CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.outdir, f"summaries_{ts}.csv")
    save_csv(overall, csv_path)

    # Plots
    overall_bar_line(overall, os.path.join(args.outdir, "overall_bar_metrics.pdf"))
    overall_backend_usage(overall, os.path.join(args.outdir, "overall_backend_usage.pdf"))
    overall_pareto(overall, os.path.join(args.outdir, "overall_pareto_reward_vs_latency.pdf"))
    per_request_pareto(records, os.path.join(args.outdir, "per_request_pareto_reward_vs_latency.pdf"))
    bytag_bar_line(by_method_tag, os.path.join(args.outdir, "bytag_latency_reward.pdf"))
    bytag_backend_usage(by_method_tag, os.path.join(args.outdir, "bytag_backend_usage.pdf"))

    # Tables
    make_pretty_table(
        overall,
        out_md=os.path.join(args.outdir, "comparison_table.md"),
        out_tex=os.path.join(args.outdir, "comparison_table.tex"),
    )

    # Console summary
    print(f"✅ Wrote {csv_path}")
    print(f"✅ Wrote PDFs to {args.outdir}/:")
    print("   - overall_bar_metrics.pdf")
    print("   - overall_backend_usage.pdf")
    print("   - overall_pareto_reward_vs_latency.pdf")
    print("   - per_request_pareto_reward_vs_latency.pdf")
    print("   - bytag_latency_reward.pdf")
    print("   - bytag_backend_usage.pdf")
    print("   - comparison_table.md / comparison_table.tex")

    print("\n=== Summary ===")
    for m in ("epsilon","ppo"):
        if m not in overall: continue
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
        print("\n=== Headline ===")
        print(f"Latency winner : {lat_winner} (lower is better)")
        print(f"Reward winner  : {rew_winner} (higher is better)")
        print(f"Quality winner : {qual_winner} (higher is better)")
        print(f"Cost winner    : {cost_winner} (lower is better)")

if __name__ == "__main__":
    main()
