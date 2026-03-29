#!/usr/bin/env python3
import argparse, glob, json, os, sys, statistics as st, math, csv, re
from collections import defaultdict
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ====== Global plotting style ======
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 16,
    'axes.labelweight': 'bold',
    'axes.titlepad': 0,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

ALL_METHODS = ['epsilon', 'ppo', 'softmax', 'ucb', 'thompson', 'moucb']
METHOD_COLORS = {
    'epsilon': '#ff7f0e',
    'ppo': '#1f77b4',
    'softmax': '#9467bd',
    'ucb': '#2ca02c',
    'thompson': '#b8860b',
    'moucb': '#d62728',
}
METHOD_HATCHES = {
    'epsilon': '//',
    'ppo': '\\\\',
    'softmax': 'xx',
    'ucb': '-',
    'thompson': '++',
    'moucb': 'oo',
}
METHOD_MARKERS = {
    'epsilon': 's',
    'ppo': 'o',
    'softmax': '^',
    'ucb': 'D',
    'thompson': 'X',
    'moucb': 'P',
}
GRID_KW = dict(linestyle='--', alpha=0.35, linewidth=0.8)
BAR_KW = dict(edgecolor='black', linewidth=0.6)
EXPECTED_REWARD = (1.0, 1.0, 0.001)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def canonical_method_name(x):
    x = (x or '').lower()
    if 'epsilon' in x:
        return 'epsilon'
    if 'softmax' in x:
        return 'softmax'
    if 'thompson' in x:
        return 'thompson'
    if 'moucb' in x or 'mo_ucb' in x or 'vector_ucb' in x or 'multiobjectiveucb' in x:
        return 'moucb'
    if x == 'ucb' or re.search(r'(^|_)ucb($|_)', x):
        return 'ucb'
    if 'ppo' in x:
        return 'ppo'
    return x or 'unknown'


def infer_run_id(path):
    base = os.path.basename(path)
    stem = base[:-6] if base.endswith('.jsonl') else base
    m = re.search(r'_(\d{8}_\d{6})$', stem)
    if m:
        return m.group(1)
    parts = stem.split('_')
    if len(parts) >= 3:
        return parts[-1]
    return stem


def infer_tag(path):
    base = os.path.basename(path).lower()
    for tag in ('code', 'explain', 'tests'):
        if f'_{tag}_' in base or base.endswith(f'_{tag}.jsonl') or f'{tag}' in base:
            return tag
    return 'unknown'


def method_from_name(path):
    return canonical_method_name(os.path.basename(path))


def find_logs(auto_dir='logs'):
    patterns = [os.path.join(auto_dir, f'route_*_{m}_*.jsonl') for m in ALL_METHODS]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    return sorted(set(files))


def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def collect_records(files):
    recs = []
    for f in files:
        method = method_from_name(f)
        tag_from_name = infer_tag(f)
        run_id = infer_run_id(f)
        for r in load_jsonl(f):
            r = dict(r)
            r['__method__'] = method
            r['__file__'] = f
            r['__run_id__'] = run_id
            if 'backend' in r:
                r['backend'] = str(r.get('backend'))
            r['step_tag'] = (r.get('step_tag') or tag_from_name or 'unknown')
            r['latency_s'] = safe_float(r.get('latency_s'))
            r['reward'] = safe_float(r.get('reward'))
            r['quality'] = safe_float(r.get('quality'))
            r['cost'] = safe_float(r.get('cost'))
            r['alpha_quality'] = safe_float(r.get('alpha_quality'))
            r['beta_latency'] = safe_float(r.get('beta_latency'))
            r['gamma_cost'] = safe_float(r.get('gamma_cost'))
            recs.append(r)
    return recs


def pct(vals, q):
    if not vals:
        return math.nan
    return float(np.percentile(np.asarray(vals, dtype=float), q))


def summarize(records):
    vals = lambda key: [safe_float(r.get(key)) for r in records if safe_float(r.get(key)) is not None]
    lat = vals('latency_s')
    rew = vals('reward')
    qual = vals('quality')
    cost = vals('cost')
    backs = [r['backend'] for r in records if r.get('backend') is not None]
    return {
        'n': len(records),
        'mean_latency': st.mean(lat) if lat else math.nan,
        'p50_latency': pct(lat, 50),
        'p95_latency': pct(lat, 95),
        'mean_reward': st.mean(rew) if rew else math.nan,
        'mean_quality': st.mean(qual) if qual else math.nan,
        'mean_cost': st.mean(cost) if cost else math.nan,
        'backend_counts': {b: backs.count(b) for b in sorted(set(backs))},
    }


def group_by(items, keys):
    d = defaultdict(list)
    for r in items:
        d[tuple(r.get(k) for k in keys)].append(r)
    return d


def color_for(method):
    return METHOD_COLORS.get(method, '#bbbbbb')


def fmt(x, d=3):
    return f'{x:.{d}f}' if x is not None and not math.isnan(x) else '–'


# ---------- reward consistency ----------
def reward_configs_by_method(records):
    configs = defaultdict(set)
    for r in records:
        method = canonical_method_name(r.get('__method__') or r.get('policy'))
        a = safe_float(r.get('alpha_quality'))
        b = safe_float(r.get('beta_latency'))
        g = safe_float(r.get('gamma_cost'))
        if None in (a, b, g):
            continue
        configs[method].add((round(a, 10), round(b, 10), round(g, 10)))
    return dict(configs)


def save_reward_config_report(records, outpath):
    configs = reward_configs_by_method(records)
    rows = []
    with open(outpath, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['method', 'configs_seen', 'matches_expected'])
        w.writeheader()
        for method in ALL_METHODS:
            cfgs = sorted(configs.get(method, set()))
            rows.append({
                'method': method,
                'configs_seen': '; '.join(str(c) for c in cfgs) if cfgs else 'missing',
                'matches_expected': 'yes' if cfgs and all(c == EXPECTED_REWARD for c in cfgs) else ('missing' if not cfgs else 'no'),
            })
        for row in rows:
            w.writerow(row)
    return rows


# ---------- plots ----------
def overall_backend_usage(summary_map, outpath):
    methods = [m for m in ALL_METHODS if m in summary_map]
    if not methods:
        return
    backends = sorted(set().union(*[summary_map[m]['backend_counts'].keys() for m in methods]))
    x = np.arange(len(backends))
    width = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(methods):
        counts = [summary_map[m]['backend_counts'].get(b, 0) for b in backends]
        bars = ax.bar(x + (i - (len(methods) - 1) / 2) * width, counts, width, label=m, color=color_for(m), **BAR_KW)
        hatch = METHOD_HATCHES.get(m)
        if hatch:
            for b in bars:
                b.set_hatch(hatch)
    ax.set_xticks(x)
    ax.set_xticklabels(backends)
    ax.set_xlabel('Backend ID', fontweight='bold')
    ax.set_ylabel('Number of Requests', fontweight='bold')
    ax.grid(True, axis='y', **GRID_KW)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def per_request_pareto_all(records, outpath):
    methods = [m for m in ALL_METHODS if any(r.get('__method__') == m for r in records)]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for m in methods:
        xs, ys = [], []
        for r in records:
            if r.get('__method__') != m:
                continue
            lat = safe_float(r.get('latency_s'))
            rew = safe_float(r.get('reward'))
            if lat is None or rew is None:
                continue
            xs.append(lat)
            ys.append(rew)
        if not xs:
            continue
        ax.scatter(xs, ys, s=16 if m == 'ppo' else 14, alpha=0.85 if m != 'ppo' else 0.75,
                   label=m, color=color_for(m), marker=METHOD_MARKERS.get(m, 'o'), edgecolors='none')
    ax.set_xlabel('Latency (s) ↓', fontweight='bold')
    ax.set_ylabel('Reward ↑', fontweight='bold')
    ax.grid(True, **GRID_KW)
    ax.legend(title='Method', frameon=False)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def overall_pareto_from_runs(method_run_summary, outpath):
    if not method_run_summary:
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for method in ALL_METHODS:
        rows = [r for r in method_run_summary if r['method'] == method]
        if not rows:
            continue
        xs = [r['mean_latency'] for r in rows if r['mean_latency'] is not None and not math.isnan(r['mean_latency'])]
        ys = [r['mean_reward'] for r in rows if r['mean_reward'] is not None and not math.isnan(r['mean_reward'])]
        if not xs or not ys:
            continue
        ax.scatter(xs, ys, s=60, alpha=0.9, label=method, color=color_for(method), marker=METHOD_MARKERS.get(method, 'o'))
    ax.set_xlabel('Run Mean Latency (s) ↓', fontweight='bold')
    ax.set_ylabel('Run Mean Reward ↑', fontweight='bold')
    ax.grid(True, **GRID_KW)
    ax.legend(title='Method', frameon=False)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def reward_distribution(records, outpath):
    methods = [m for m in ALL_METHODS if any(r.get('__method__') == m for r in records)]
    data = {m: [] for m in methods}
    for r in records:
        m = r.get('__method__')
        rew = safe_float(r.get('reward'))
        if m in data and rew is not None:
            data[m].append(rew)
    methods_present = [m for m in methods if data[m]]
    if not methods_present:
        return
    pos = np.arange(len(methods_present))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    bp = ax.boxplot([data[m] for m in methods_present], positions=pos, widths=0.55, showfliers=False, patch_artist=True)
    for box, m in zip(bp['boxes'], methods_present):
        box.set_facecolor(color_for(m))
        box.set_alpha(0.85)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2.0)
    for i, m in enumerate(methods_present):
        ys = data[m]
        xs = np.random.normal(loc=pos[i], scale=0.05, size=len(ys))
        ax.scatter(xs, ys, s=7, alpha=0.6, color='black', edgecolors='none')
    ax.set_xticks(pos)
    ax.set_xticklabels(methods_present)
    ax.set_ylabel('Reward ↑', fontweight='bold')
    ax.grid(True, axis='y', **GRID_KW)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def latency_distribution(records, outpath):
    methods = [m for m in ALL_METHODS if any(r.get('__method__') == m for r in records)]
    data = {m: [] for m in methods}
    for r in records:
        m = r.get('__method__')
        lat = safe_float(r.get('latency_s'))
        if m in data and lat is not None:
            data[m].append(lat)
    methods_present = [m for m in methods if data[m]]
    if not methods_present:
        return
    pos = np.arange(len(methods_present))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    bp = ax.boxplot([data[m] for m in methods_present], positions=pos, widths=0.55, showfliers=False, patch_artist=True)
    for box, m in zip(bp['boxes'], methods_present):
        box.set_facecolor(color_for(m))
        box.set_alpha(0.85)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2.0)
    for i, m in enumerate(methods_present):
        ys = data[m]
        xs = np.random.normal(loc=pos[i], scale=0.05, size=len(ys))
        ax.scatter(xs, ys, s=7, alpha=0.6, color='black', edgecolors='none')
    ax.set_xticks(pos)
    ax.set_xticklabels(methods_present)
    ax.set_ylabel('Latency (s) ↓', fontweight='bold')
    ax.grid(True, axis='y', **GRID_KW)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def bytag_bar_line(by_method_tag, outpath):
    tags = [t for t in ('code', 'explain', 'tests', 'unknown') if any(k[1] == t for k in by_method_tag.keys())]
    tags = [t for t in tags if t != 'unknown' or len(tags) == 1]
    if not tags:
        return
    fig, axes = plt.subplots(1, len(tags), figsize=(6 * len(tags), 4.8), sharey=False)
    if len(tags) == 1:
        axes = [axes]
    for ax, tag in zip(axes, tags):
        methods_present = [m for m in ALL_METHODS if (m, tag) in by_method_tag]
        if not methods_present:
            ax.axis('off')
            continue
        xs, lat_vals = [], []
        for m in methods_present:
            s = summarize(by_method_tag[(m, tag)])
            xs.append(m.upper())
            lat_vals.append(s['mean_latency'])
        bars = ax.bar(xs, lat_vals, color=[color_for(m) for m in methods_present], **BAR_KW)
        for bar, m in zip(bars, methods_present):
            hatch = METHOD_HATCHES.get(m)
            if hatch:
                bar.set_hatch(hatch)
        ax.set_xlabel(f'Method ({tag})', fontweight='bold')
        ax.set_ylabel('Mean Latency (s)', fontweight='bold')
        ax.grid(True, axis='y', **GRID_KW)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def bytag_backend_usage(by_method_tag, outpath):
    tags = [t for t in ('code', 'explain', 'tests', 'unknown') if any(k[1] == t for k in by_method_tag.keys())]
    tags = [t for t in tags if t != 'unknown' or len(tags) == 1]
    if not tags:
        return
    fig, axes = plt.subplots(1, len(tags), figsize=(6 * len(tags), 4.8), sharey=False)
    if len(tags) == 1:
        axes = [axes]
    for ax, tag in zip(axes, tags):
        methods = [m for m in ALL_METHODS if (m, tag) in by_method_tag]
        if not methods:
            ax.axis('off')
            continue
        backs_union = set()
        for m in methods:
            backs_union |= set(summarize(by_method_tag[(m, tag)])['backend_counts'].keys())
        backs = sorted(backs_union)
        x = np.arange(len(backs))
        width = 0.8 / max(len(methods), 1)
        for i, m in enumerate(methods):
            s = summarize(by_method_tag[(m, tag)])
            counts = [s['backend_counts'].get(b, 0) for b in backs]
            bars = ax.bar(x + (i - (len(methods) - 1) / 2) * width, counts, width, color=color_for(m), label=m, **BAR_KW)
            hatch = METHOD_HATCHES.get(m)
            if hatch:
                for b in bars:
                    b.set_hatch(hatch)
        ax.set_xticks(x)
        ax.set_xticklabels(backs)
        ax.set_xlabel(f'Backend ({tag})', fontweight='bold')
        ax.set_ylabel('Count', fontweight='bold')
        ax.grid(True, axis='y', **GRID_KW)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)


# ---------- tables and stats ----------
def save_csv(summary_map, out_csv):
    fields = ['method', 'n', 'mean_latency', 'p50_latency', 'p95_latency', 'mean_reward', 'mean_quality', 'mean_cost', 'backend_counts']
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for method in ALL_METHODS:
            if method not in summary_map:
                continue
            row = {'method': method}
            row.update(summary_map[method])
            w.writerow(row)


def bootstrap_ci(vals, n_boot=5000, alpha=0.05, seed=7):
    vals = [float(v) for v in vals if v is not None and not math.isnan(v)]
    if not vals:
        return math.nan, math.nan
    if len(vals) == 1:
        return vals[0], vals[0]
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=float)
    boot = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return lo, hi


def permutation_pvalue(a, b, n_perm=20000, seed=11):
    a = np.asarray([x for x in a if x is not None and not math.isnan(x)], dtype=float)
    b = np.asarray([x for x in b if x is not None and not math.isnan(x)], dtype=float)
    if len(a) < 2 or len(b) < 2:
        return math.nan
    observed = abs(a.mean() - b.mean())
    combined = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        a_star = combined[:len(a)]
        b_star = combined[len(a):]
        if abs(a_star.mean() - b_star.mean()) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def build_run_level_summary(records):
    by_run = group_by(records, ['__method__', '__run_id__', 'step_tag'])
    rows = []
    for (method, run_id, step_tag), recs in by_run.items():
        s = summarize(recs)
        rows.append({
            'method': method,
            'run_id': run_id,
            'step_tag': step_tag,
            'n': s['n'],
            'mean_latency': s['mean_latency'],
            'p50_latency': s['p50_latency'],
            'p95_latency': s['p95_latency'],
            'mean_reward': s['mean_reward'],
            'mean_quality': s['mean_quality'],
            'mean_cost': s['mean_cost'],
        })
    # overall per run pooled across tags/files for stronger stats
    by_run_overall = group_by(records, ['__method__', '__run_id__'])
    for (method, run_id), recs in by_run_overall.items():
        s = summarize(recs)
        rows.append({
            'method': method,
            'run_id': run_id,
            'step_tag': 'all',
            'n': s['n'],
            'mean_latency': s['mean_latency'],
            'p50_latency': s['p50_latency'],
            'p95_latency': s['p95_latency'],
            'mean_reward': s['mean_reward'],
            'mean_quality': s['mean_quality'],
            'mean_cost': s['mean_cost'],
        })
    rows.sort(key=lambda r: (r['step_tag'], ALL_METHODS.index(r['method']) if r['method'] in ALL_METHODS else 999, str(r['run_id'])))
    return rows


def save_run_level_summary(rows, out_csv):
    fields = ['method', 'run_id', 'step_tag', 'n', 'mean_latency', 'p50_latency', 'p95_latency', 'mean_reward', 'mean_quality', 'mean_cost']
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_method_ci_summary(run_rows):
    out = []
    tags = sorted(set(r['step_tag'] for r in run_rows))
    metrics = ['mean_latency', 'mean_reward', 'mean_quality', 'mean_cost']
    for tag in tags:
        for method in ALL_METHODS:
            rows = [r for r in run_rows if r['step_tag'] == tag and r['method'] == method]
            if not rows:
                continue
            row = {'step_tag': tag, 'method': method, 'runs': len(rows)}
            for metric in metrics:
                vals = [r[metric] for r in rows]
                mean = float(np.mean(vals)) if vals else math.nan
                lo, hi = bootstrap_ci(vals)
                row[metric] = mean
                row[f'{metric}_ci_low'] = lo
                row[f'{metric}_ci_high'] = hi
            out.append(row)
    return out


def save_method_ci_summary(rows, out_csv):
    if not rows:
        return
    fields = ['step_tag', 'method', 'runs',
              'mean_latency', 'mean_latency_ci_low', 'mean_latency_ci_high',
              'mean_reward', 'mean_reward_ci_low', 'mean_reward_ci_high',
              'mean_quality', 'mean_quality_ci_low', 'mean_quality_ci_high',
              'mean_cost', 'mean_cost_ci_low', 'mean_cost_ci_high']
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_significance_table(run_rows, baseline='ppo'):
    results = []
    metrics = {
        'mean_latency': 'lower',
        'mean_reward': 'higher',
        'mean_quality': 'higher',
        'mean_cost': 'lower',
    }
    tags = sorted(set(r['step_tag'] for r in run_rows))
    for tag in tags:
        base_rows = [r for r in run_rows if r['step_tag'] == tag and r['method'] == baseline]
        if not base_rows:
            continue
        for method in ALL_METHODS:
            if method == baseline:
                continue
            cmp_rows = [r for r in run_rows if r['step_tag'] == tag and r['method'] == method]
            if not cmp_rows:
                continue
            row = {'step_tag': tag, 'baseline': baseline, 'comparison': method,
                   'baseline_runs': len(base_rows), 'comparison_runs': len(cmp_rows)}
            for metric, direction in metrics.items():
                a = [r[metric] for r in base_rows]
                b = [r[metric] for r in cmp_rows]
                p = permutation_pvalue(a, b)
                diff = (np.mean(a) - np.mean(b)) if a and b else math.nan
                # orient positive as PPO advantage
                if direction == 'lower':
                    diff = -diff
                row[f'{metric}_pvalue'] = p
                row[f'{metric}_ppo_advantage'] = diff
            results.append(row)
    return results


def save_significance_table(rows, out_csv):
    if not rows:
        return
    fields = ['step_tag', 'baseline', 'comparison', 'baseline_runs', 'comparison_runs',
              'mean_latency_pvalue', 'mean_latency_ppo_advantage',
              'mean_reward_pvalue', 'mean_reward_ppo_advantage',
              'mean_quality_pvalue', 'mean_quality_ppo_advantage',
              'mean_cost_pvalue', 'mean_cost_ppo_advantage']
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def make_pretty_table(overall, out_md, out_tex):
    methods = [m for m in ALL_METHODS if m in overall]
    if not methods:
        return
    header = (
        '# Overall Routing Performance by Method\n\n'
        '| Method | n | Mean Latency (s) | p50 (s) | p95 (s) | Mean Reward | Mean Quality | Mean Cost |\n'
        '|:--|--:|--:|--:|--:|--:|--:|--:|\n'
    )
    rows = []
    for m in methods:
        s = overall[m]
        rows.append(f"| {m} | {s['n']} | {fmt(s['mean_latency'])} | {fmt(s['p50_latency'])} | {fmt(s['p95_latency'])} | {fmt(s['mean_reward'])} | {fmt(s['mean_quality'])} | {fmt(s['mean_cost'])} |")
    md = header + '\n'.join(rows) + '\n'
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(md)
    tex_lines = [
        r'\begin{table}[t]', r'\centering', r'\small', r'\begin{tabular}{lrrrrrrr}', r'\toprule',
        r'Method & $n$ & Mean Lat. & p50 & p95 & Mean Rew. & Mean Qual. & Mean Cost \\', r'\midrule'
    ]
    for m in methods:
        s = overall[m]
        tex_lines.append(f"{m} & {s['n']} & {fmt(s['mean_latency'])} & {fmt(s['p50_latency'])} & {fmt(s['p95_latency'])} & {fmt(s['mean_reward'])} & {fmt(s['mean_quality'])} & {fmt(s['mean_cost'])} " + r"\\")
    tex_lines.extend([r'\bottomrule', r'\end{tabular}',
                      r'\caption{Overall routing performance across all methods. Lower latency and cost, and higher reward and quality are desirable.}',
                      r'\end{table}', ''])
    with open(out_tex, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex_lines))


def winner_name(metric_ppo, metric_eps, higher_is_better):
    if any(v is None or math.isnan(v) for v in (metric_ppo, metric_eps)):
        return 'tie'
    if higher_is_better:
        return 'ppo' if metric_ppo > metric_eps else ('epsilon' if metric_eps > metric_ppo else 'tie')
    return 'ppo' if metric_ppo < metric_eps else ('epsilon' if metric_eps < metric_ppo else 'tie')


def main():
    ap = argparse.ArgumentParser(description='Analyze router JSONL logs.')
    ap.add_argument('--outdir', default='results', help='Where to write CSV/PDF outputs.')
    ap.add_argument('--logdir', default='logs', help='Directory containing router logs.')
    args = ap.parse_args()

    files = find_logs(args.logdir)
    if not files:
        print(f'No log files found under ./{args.logdir}/')
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    records = collect_records(files)
    if not records:
        print('No records parsed.')
        sys.exit(2)

    by_method = group_by(records, ['__method__'])
    overall = {k[0]: summarize(v) for k, v in by_method.items()}
    by_method_tag = group_by(records, ['__method__', 'step_tag'])
    run_rows = build_run_level_summary(records)
    ci_rows = build_method_ci_summary(run_rows)
    sig_rows = build_significance_table(run_rows, baseline='ppo')

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_csv(overall, os.path.join(args.outdir, f'summaries_{ts}.csv'))
    save_run_level_summary(run_rows, os.path.join(args.outdir, 'run_level_summary.csv'))
    save_method_ci_summary(ci_rows, os.path.join(args.outdir, 'method_summary_with_ci.csv'))
    save_significance_table(sig_rows, os.path.join(args.outdir, 'ppo_pairwise_significance.csv'))
    reward_rows = save_reward_config_report(records, os.path.join(args.outdir, 'reward_config_report.csv'))

    overall_backend_usage(overall, os.path.join(args.outdir, 'overall_backend_usage.pdf'))
    overall_pareto_from_runs([r for r in run_rows if r['step_tag'] == 'all'], os.path.join(args.outdir, 'overall_pareto_reward_vs_latency.pdf'))
    per_request_pareto_all(records, os.path.join(args.outdir, 'per_request_pareto_all_methods.pdf'))
    reward_distribution(records, os.path.join(args.outdir, 'reward_distribution_by_method.pdf'))
    latency_distribution(records, os.path.join(args.outdir, 'latency_distribution_by_method.pdf'))
    bytag_bar_line(by_method_tag, os.path.join(args.outdir, 'bytag_latency_reward.pdf'))
    bytag_backend_usage(by_method_tag, os.path.join(args.outdir, 'bytag_backend_usage.pdf'))
    make_pretty_table(overall, os.path.join(args.outdir, 'comparison_table.md'), os.path.join(args.outdir, 'comparison_table.tex'))

    print(f'Wrote outputs to {args.outdir}/')
    print('   - overall_backend_usage.pdf')
    print('   - overall_pareto_reward_vs_latency.pdf')
    print('   - per_request_pareto_all_methods.pdf')
    print('   - reward_distribution_by_method.pdf')
    print('   - latency_distribution_by_method.pdf')
    print('   - bytag_latency_reward.pdf')
    print('   - bytag_backend_usage.pdf')
    print('   - comparison_table.md / comparison_table.tex')
    print('   - run_level_summary.csv')
    print('   - method_summary_with_ci.csv')
    print('   - ppo_pairwise_significance.csv')
    print('   - reward_config_report.csv')

    print('\n=== Reward configuration check ===')
    for row in reward_rows:
        print(f"[{row['method']}] configs={row['configs_seen']} expected={EXPECTED_REWARD} match={row['matches_expected']}")

    print('\n=== Summary ===')
    for m in [m for m in ALL_METHODS if m in overall]:
        s = overall[m]
        print(f"[{m}] n={s['n']}")
        print(f"  mean latency: {s['mean_latency']:.3f}s   p50: {s['p50_latency']:.3f}s   p95: {s['p95_latency']:.3f}s")
        print(f"  mean reward : {s['mean_reward']:.3f}    mean quality: {s['mean_quality']:.3f}")
        print(f"  mean cost   : {s['mean_cost']:.3f}")
        print(f"  backend counts: {s['backend_counts']}")

    if 'ppo' in overall and 'epsilon' in overall:
        print('\n=== Headline (PPO vs epsilon) ===')
        print(f"Latency winner : {winner_name(overall['ppo']['mean_latency'], overall['epsilon']['mean_latency'], False)} (lower is better)")
        print(f"Reward winner  : {winner_name(overall['ppo']['mean_reward'], overall['epsilon']['mean_reward'], True)} (higher is better)")
        print(f"Quality winner : {winner_name(overall['ppo']['mean_quality'], overall['epsilon']['mean_quality'], True)} (higher is better)")
        print(f"Cost winner    : {winner_name(overall['ppo']['mean_cost'], overall['epsilon']['mean_cost'], False)} (lower is better)")

if __name__ == '__main__':
    main()
