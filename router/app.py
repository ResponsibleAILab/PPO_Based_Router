# =============================
# File: router/app.py  (3-LLM version)
# =============================
from __future__ import annotations
import os, time
from typing import List, Dict
import httpx

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

from policy import EpsilonGreedyPolicy, PPOPolicy
from metrics import SlidingWindow

# ── Config via env
POLICY_KIND = os.getenv("POLICY", "ppo").lower()          # "ppo" or "epsilon"
EPSILON = float(os.getenv("EPSILON", "0.1"))
WINDOW = int(os.getenv("METRICS_WINDOW", "50"))
BACKEND_TARGETS = os.getenv("BACKEND_TARGETS", "").strip()

def _parse_backends() -> List[str]:
    if BACKEND_TARGETS:
        urls = [u.strip() for u in BACKEND_TARGETS.split(",") if u.strip()]
        if urls:
            return urls
    # Fallback (useful if env not set)
    return ["http://mcp_a:8000", "http://mcp_b:8000", "http://mcp_c:8000"]

BACKENDS: List[str] = _parse_backends()
N_ACTIONS = len(BACKENDS)                                  # action space = number of backends

# ── Metrics
REQS = Counter("router_requests_total", "Total requests")
ROUTED = Counter("router_routed_total", "Total routed to backend", ["backend"])
FAILS = Counter("router_failures_total", "Total routing failures", ["reason"])
LAT = Histogram("router_end_to_end_latency_seconds", "End-to-end latency (s)")
BACKEND_LAT = Histogram("backend_latency_seconds", "Observed backend latency (s)", ["backend"])
QUEUE = Gauge("backend_estimated_queue", "Estimated backend queue length", ["backend"])

# ── App & policy
app = FastAPI(title="RL Router (3 LLMs)")
policy = PPOPolicy(n_slots=N_ACTIONS, obs_dim=6) if POLICY_KIND == "ppo" else EpsilonGreedyPolicy(epsilon=EPSILON)
windows: Dict[int, SlidingWindow] = {i: SlidingWindow(WINDOW) for i in range(N_ACTIONS)}

@app.get("/health")
def health():
    return {"status": "ok", "policy": POLICY_KIND, "backends": BACKENDS}

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/infer")
async def infer(payload: Dict):
    REQS.inc()

    # Build observation per backend (slot i)
    obs = []
    for i in range(N_ACTIONS):
        m = windows[i]
        mean = m.mean() or 0.2
        p95 = m.percentile(95) or 0.4
        q = m.queue_estimate()
        obs.append([mean, p95, q])
        QUEUE.labels(backend=str(i)).set(q)

    # Policy chooses one of the 3 backends
    choice = policy.select(obs, valid_n=N_ACTIONS) if hasattr(policy, "select") else 0
    target = BACKENDS[choice]

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
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

    # Reward: negative latency (extend later with quality/cost)
    reward = -dt
    if hasattr(policy, "observe"):
        policy.observe(reward)

    return {"backend": choice, "backend_url": target, "latency_s": dt, "result": out}
