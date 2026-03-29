# =============================
# File: router/app.py  (3-LLM + step-aware PPO + checkpoints + multi-objective reward)
# =============================
from __future__ import annotations
import os, time, json
from typing import List, Dict
import httpx

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

from policy import EpsilonGreedyPolicy, PPOPolicy, SoftmaxPolicy, UCBPolicy, ThompsonSamplingPolicy, MultiObjectiveUCBPolicy
from metrics import SlidingWindow

# ── Config via env
POLICY_KIND = os.getenv("POLICY", "ppo").lower()          # "ppo" or "epsilon"
POLICY_NAME = POLICY_KIND
EPSILON = float(os.getenv("EPSILON", "0.1"))
WINDOW = int(os.getenv("METRICS_WINDOW", "50"))
BACKEND_TARGETS = os.getenv("BACKEND_TARGETS", "").strip()
CKPT = os.getenv("CHECKPOINT_PATH", "/app/checkpoints/ppo.pt")
SAVE_EVERY = int(os.getenv("SAVE_EVERY_UPDATES", "1"))

# Optional run metadata (passed through from planner, but we also read defaults here)
RUN_ID_DEFAULT = os.getenv("RUN_ID", "run-unknown")
DATASET_TAG_DEFAULT = os.getenv("DATASET_TAG", "dataset-unknown")

# Backend names (for pretty logs)
BACKEND_NAMES = ["mcp_a","mcp_b","mcp_c"]

# Step tags (used for one-hot feature appended to obs)
STEP_TAGS_ENV = os.getenv("STEP_TAGS", "code,explain,tests")
STEP_TAGS = [t.strip().lower() for t in STEP_TAGS_ENV.split(",") if t.strip()]
OBS_BASE = 3  # [mean, p95, queue]   (per-backend perf features)

# Multi-objective reward weights + logging
ALPHA = float(os.getenv("ALPHA_QUALITY", "1.0"))
BETA  = float(os.getenv("BETA_LAT", "1.0"))
GAMMA = float(os.getenv("GAMMA_COST", "0.0"))
COSTS = [float(x) for x in os.getenv("COSTS_PER_TOKEN", "0.0,0.0,0.0").split(",")]

# If ROUTER_JSONL is set, honor it; otherwise auto-name by policy
_log_env = os.getenv("ROUTER_JSONL")
LOG_PATH = _log_env if _log_env else f"/app/logs/route_log_{POLICY_NAME}.jsonl"


def _reward_components(quality: float, latency_s: float, cost: float) -> Dict[str, float]:
    quality_term = ALPHA * float(quality)
    latency_term = BETA * float(latency_s)
    cost_term = GAMMA * float(cost)
    return {
        "quality_term": quality_term,
        "latency_term": latency_term,
        "cost_term": cost_term,
        "reward": quality_term - latency_term - cost_term,
    }

def _quality_score(step_tag: str | None, prompt: str, result: dict) -> float:
    """Very simple, deterministic heuristics (0..1). Refine later."""
    text = (result or {}).get("echo") or (result or {}).get("response") or (result or {}).get("text") or ""
    t = (step_tag or "").lower()
    if t == "tests":
        hits = 0
        hits += text.count("assert")
        hits += text.count("test_")
        return min(1.0, hits / 3.0)
    if t == "code":
        score = 0.0
        if "def " in text: score += 0.5
        if ":" in text and "(" in text and ")" in text: score += 0.2
        if len(text) > 60: score += 0.3
        return min(1.0, score)
    # explain/default: favor clearer, longer explanations (very rough)
    L = len(text.split())
    return max(0.0, min(1.0, (L - 20) / 100.0))  # 0 at 20 words → ~1 at 120+

def _parse_backends() -> List[str]:
    if BACKEND_TARGETS:
        urls = [u.strip() for u in BACKEND_TARGETS.split(",") if u.strip()]
        if urls:
            return urls
    return ["http://mcp_a:8000", "http://mcp_b:8000", "http://mcp_c:8000"]

BACKENDS: List[str] = _parse_backends()
N_ACTIONS = len(BACKENDS)

def one_hot_step(tag: str | None) -> List[float]:
    v = [0.0] * len(STEP_TAGS)
    if not tag:
        return v
    try:
        i = STEP_TAGS.index(tag.lower())
        v[i] = 1.0
    except ValueError:
        pass
    return v

# ── Metrics
REQS = Counter("router_requests_total", "Total requests")
ROUTED = Counter("router_routed_total", "Total routed to backend", ["backend"])
FAILS = Counter("router_failures_total", "Total routing failures", ["reason"])
LAT = Histogram("router_end_to_end_latency_seconds", "End-to-end latency (s)")
BACKEND_LAT = Histogram("backend_latency_seconds", "Observed backend latency (s)", ["backend"])
QUEUE = Gauge("backend_estimated_queue", "Estimated backend queue length", ["backend"])

# ── App & policy
app = FastAPI(title="RL Router (3 LLMs, step-aware PPO)")

# obs_dim = perf(3) pooled to 6 (min/mean per feat) + onehot(len(STEP_TAGS)) + context(3)
# NOTE: PPOPolicy will auto-adapt obs_dim if it changes at runtime.
INITIAL_OBS_DIM = 6 + len(STEP_TAGS) + 3

if POLICY_KIND == "ppo":
    policy = PPOPolicy(n_slots=N_ACTIONS, obs_dim=INITIAL_OBS_DIM)
elif POLICY_KIND == "epsilon":
    policy = EpsilonGreedyPolicy(epsilon=EPSILON)
elif POLICY_KIND == "softmax":
    policy = SoftmaxPolicy(temperature=float(os.getenv("SOFTMAX_TAU", "0.1")))
elif POLICY_KIND == "ucb":
    policy = UCBPolicy(c=float(os.getenv("UCB_C", "2.0")))
elif POLICY_KIND in ("moucb", "mo_ucb", "multiobjective_ucb", "multi_objective_ucb", "vector_ucb"):
    policy = MultiObjectiveUCBPolicy(
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA,
        c=float(os.getenv("MOUCB_C", os.getenv("UCB_C", "2.0"))),
    )
elif POLICY_KIND in ("thompson", "ts", "thompson_sampling"):
    policy = ThompsonSamplingPolicy(prior_var=float(os.getenv("TS_PRIOR_VAR", "1.0")))
else:
    raise RuntimeError(f"Unknown POLICY '{POLICY_KIND}'")

# Sliding latency windows per logical backend index
windows: Dict[int, SlidingWindow] = {i: SlidingWindow(WINDOW) for i in range(N_ACTIONS)}

# Try to autoload a checkpoint at startup (if PPO)
if POLICY_KIND == "ppo":
    try:
        if os.path.exists(CKPT):
            policy.load(CKPT)
            print(f"[router] Loaded PPO policy from {CKPT}")
    except Exception as e:
        print(f"[router] PPO load failed: {e}")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "policy": POLICY_KIND,
        "backends": BACKENDS,
        "backend_names": BACKEND_NAMES[:N_ACTIONS],
        "step_tags": STEP_TAGS,
        "checkpoint_path": CKPT,
        "save_every_updates": SAVE_EVERY,
        "alpha_quality": ALPHA,
        "beta_latency": BETA,
        "gamma_cost": GAMMA,
        "log_path": LOG_PATH,
    }

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/policy/save")
def save_policy():
    if POLICY_KIND != "ppo":
        raise HTTPException(status_code=400, detail="Policy is not PPO.")
    try:
        policy.save(CKPT)
        return {"status": "ok", "saved_to": CKPT}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/policy/load")
def load_policy():
    if POLICY_KIND != "ppo":
        raise HTTPException(status_code=400, detail="Policy is not PPO.")
    try:
        policy.load(CKPT)
        return {"status": "ok", "loaded_from": CKPT}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/infer")
async def infer(payload: Dict):
    REQS.inc()

    # Inputs (with safe fallbacks)
    step_tag = (payload or {}).get("step_tag")
    prompt_in = (payload or {}).get("prompt", "")
    run_id = (payload or {}).get("run_id") or RUN_ID_DEFAULT
    dataset_tag = (payload or {}).get("dataset_tag") or DATASET_TAG_DEFAULT
    step_id = (payload or {}).get("step_id")

    # Context features (identical across backends; appended to each row)
    step_len = len(prompt_in)
    step_len_norm = min(1.0, step_len / 2000.0)  # [0..1]
    pi = prompt_in.lower()
    contains_code_kw = 1.0 if any(k in pi for k in ["def ", "class ", "function", "snippet"]) else 0.0
    contains_test_kw = 1.0 if any(k in pi for k in ["assert", "test_"]) else 0.0

    # Observation per backend
    obs = []
    oh = one_hot_step(step_tag)  # compute once
    for i in range(N_ACTIONS):
        m = windows[i]
        mean = m.mean() or 0.2
        p95 = m.percentile(95) or 0.4
        q = m.queue_estimate()
        row = [mean, p95, q] + oh + [step_len_norm, contains_code_kw, contains_test_kw]
        obs.append(row)
        QUEUE.labels(backend=str(i)).set(q)

    # Policy chooses one of the backends
    if POLICY_KIND == "ppo":
        choice = policy.select(obs, valid_n=N_ACTIONS)
    else:
        choice = policy.select(obs)

    target = BACKENDS[choice]
    backend_name = BACKEND_NAMES[choice] if choice < len(BACKEND_NAMES) else str(choice)

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            r = await client.post(f"{target}/infer", json=payload)
            r.raise_for_status()
            out = r.json()
    except Exception as e:
        FAILS.labels(reason=type(e).__name__).inc()
        raise HTTPException(status_code=502, detail=f"Backend error: {e}")
    finally:
        dt = time.perf_counter() - t0
        LAT.observe(dt)
        BACKEND_LAT.labels(backend=str(choice)).observe(dt)
        windows[choice].append(dt)
        ROUTED.labels(backend=str(choice)).inc()

    # Tokens (adapter returns; else estimate)
    tokens = (out or {}).get("tokens")
    if tokens is None:
        tokens = max(1, len(((out or {}).get("echo") or "").split()))
    per_token = COSTS[choice] if choice < len(COSTS) else 0.0
    cost = per_token * float(tokens)

    # Quality + multi-objective reward
    quality = _quality_score(step_tag, prompt_in, out)
    comps = _reward_components(quality, dt, cost)
    reward = comps["reward"]

    # Update learning policy (PPO) or bandit policy
    if POLICY_KIND == "ppo" and hasattr(policy, "observe"):
        policy.observe(reward)
    elif hasattr(policy, "update_multi"):
        try:
            policy.update_multi(choice, quality, dt, cost)
        except TypeError:
            pass
    elif hasattr(policy, "update"):
        # bandit-style policies (softmax / ucb / thompson / epsilon)
        try:
            policy.update(choice, reward)
        except TypeError:
            # in case a policy has a different signature; we ignore gracefully
            pass

    # JSONL log (for analysis/paper)
    rec = {
        "ts": time.time(),
        "policy": POLICY_NAME,
        "backend": choice,
        "backend_name": backend_name,
        "backend_url": target,
        "model_name": (out or {}).get("model"),
        "latency_s": dt,
        "tokens": tokens,
        "cost": cost,
        "quality": quality,
        "reward": reward,
        "reward_quality_term": comps["quality_term"],
        "reward_latency_term": comps["latency_term"],
        "reward_cost_term": comps["cost_term"],
        "alpha_quality": ALPHA,
        "beta_latency": BETA,
        "gamma_cost": GAMMA,
        "cost_per_token": per_token,
        "step_tag": step_tag,
        "step_id": step_id,
        "prompt_len": step_len,
        "run_id": run_id,
        "dataset_tag": dataset_tag,
    }
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

    return {
        "backend": choice,
        "backend_url": target,
        "latency_s": dt,
        "step_tag": step_tag,
        "result": out
    }
