#!/usr/bin/env python3
import os, glob, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results/plots", exist_ok=True)

# Load combined rows from logs (same as analysis)
rows = []
for f in glob.glob("logs/route_*_{ppo,epsilon}_*.jsonl"):
    stem = Path(f).stem
    parts = stem.split("_")
    dataset = parts[1] if len(parts)>=4 else "unknown"
    method = parts[2] if len(parts)>=4 else ("ppo" if "ppo" in stem else "epsilon")
    for line in open(f, "r", encoding="utf-8"):
        try:
            j = json.loads(line)
            j["dataset"], j["method"], j["file"] = dataset, method, stem
            rows.append(j)
        except:
            pass

df = pd.DataFrame(rows)
for col in ["latency_s","quality","reward","backend"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 1) Reward vs Latency (scatter)
for ds, sub in df.groupby("dataset"):
    plt.figure(figsize=(6,4))
    for m, ss in sub.groupby("method"):
        plt.scatter(ss["latency_s"], ss["reward"], s=12, alpha=0.6, label=m)
    plt.xlabel("Latency (s)")
    plt.ylabel("Reward")
    plt.title(f"{ds}: Reward vs Latency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/plots/{ds}_reward_vs_latency.pdf")
    plt.close()

# 2) Backend distribution (stacked bars)
for ds, sub in df.groupby("dataset"):
    ct = pd.crosstab(sub["method"], sub["backend"], normalize="index")
    ct.plot(kind="bar", stacked=True, figsize=(6,4))
    plt.ylabel("Proportion")
    plt.title(f"{ds}: Routing distribution by backend")
    plt.tight_layout()
    plt.savefig(f"results/plots/{ds}_routing_distribution.pdf")
    plt.close()

# 3) Quality vs Latency with error bars
for ds, sub in df.groupby("dataset"):
    agg = sub.groupby("method").agg(
        mean_quality=("quality","mean"),
        sd_quality=("quality","std"),
        mean_latency=("latency_s","mean"),
        sd_latency=("latency_s","std"),
    ).reset_index()
    plt.figure(figsize=(6,4))
    plt.errorbar(agg["mean_latency"], agg["mean_quality"],
                 xerr=agg["sd_latency"], yerr=agg["sd_quality"],
                 fmt="o")
    for _, r in agg.iterrows():
        plt.annotate(r["method"], (r["mean_latency"], r["mean_quality"]),
                     textcoords="offset points", xytext=(5,5))
    plt.xlabel("Mean Latency (s) ± sd")
    plt.ylabel("Mean Quality ± sd")
    plt.title(f"{ds}: Quality vs Latency")
    plt.tight_layout()
    plt.savefig(f"results/plots/{ds}_quality_vs_latency.pdf")
    plt.close()

print("Wrote PDFs under results/plots/")
