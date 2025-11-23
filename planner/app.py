from __future__ import annotations
import os, re
from typing import List, Dict, Optional
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROUTER_URL = os.getenv("ROUTER_URL", "http://rl_router:8080")
# Keep these aligned with router STEP_TAGS (default: "code,explain,tests")
ROUTER_STEP_TAGS = [t.strip().lower() for t in os.getenv("STEP_TAGS", "code,explain,tests").split(",") if t.strip()]
RUN_ID = os.getenv("RUN_ID", "run-unknown")
DATASET_TAG = os.getenv("DATASET_TAG", "dataset-unknown")

app = FastAPI(title="Planner → RL Router")

# ── Schemas ───────────────────────────────────────────────────────────────────
class PlanIn(BaseModel):
    prompt: str
    max_new_tokens: int | None = 256

class Subtask(BaseModel):
    id: int
    type: str
    text: str
    depends_on: Optional[int] = None

class PlanOut(BaseModel):
    subtasks: List[Subtask]

class RunOut(BaseModel):
    steps: List[Dict]
    final_answer: str

# ── Simple rule-based planner ─────────────────────────────────────────────────
BULLET = re.compile(r"^\s*(?:[-*•]|\d+[\.)])\s+")
CODE_HINT = re.compile(r"\b(code|python|function|class|implement|snippet)\b", re.I)
MATH_HINT = re.compile(r"\b(math|equation|algebra|derive|proof|calculate|compute)\b", re.I)
TEST_HINT = re.compile(r"\b(unit\s*test|tests?)\b", re.I)
EXPLAIN_HINT = re.compile(r"\b(explain|summarize|summary|overview|document|comment|describe|why)\b", re.I)

def split_lines(prompt: str) -> List[str]:
    lines = [ln.strip() for ln in prompt.strip().splitlines() if ln.strip()]
    # If no bullets/numbers, treat the whole prompt as one task
    if not any(BULLET.search(ln) for ln in lines):
        return [prompt.strip()]
    # Strip bullet/number prefixes
    cleaned = [BULLET.sub("", ln) for ln in lines]
    return cleaned

def classify(text: str) -> str:
    t = text.lower()
    if TEST_HINT.search(t): return "tests"    # normalize to router tag
    if CODE_HINT.search(t): return "code"
    if MATH_HINT.search(t): return "explain"  # route math to reasoning/explain by default
    if EXPLAIN_HINT.search(t): return "explain"
    return "explain"  # default to explain; safer than "general" for now

def plan(prompt: str) -> List[Subtask]:
    items = split_lines(prompt)
    subtasks: List[Subtask] = []
    last_code_id: Optional[int] = None
    for i, txt in enumerate(items, start=1):
        t = classify(txt)
        dep = None
        if t == "tests" and last_code_id is not None:
            dep = last_code_id
        st = Subtask(id=i, type=t, text=txt, depends_on=dep)
        subtasks.append(st)
        if t == "code":
            last_code_id = i
    return subtasks

# ── Execution engine: run subtasks through RL router ──────────────────────────
async def call_router_infer(text: str, max_new_tokens: int, step_tag: str | None, step_id: int | None) -> Dict:
    # ensure step_tag matches router’s known tags (otherwise send None)
    tag = step_tag if (step_tag and step_tag.lower() in ROUTER_STEP_TAGS) else None
    payload = {
        "prompt": text,
        "max_new_tokens": max_new_tokens,
        "step_tag": tag,
        "run_id": RUN_ID,
        "dataset_tag": DATASET_TAG,
        "step_id": step_id,
    }
    async with httpx.AsyncClient(timeout=300.0) as client:
        r = await client.post(f"{ROUTER_URL}/infer", json=payload)
        try:
            r.raise_for_status()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Router error: {e} / {r.text[:200]}")
        return r.json()

def topo_order(subtasks: List[Subtask]) -> List[Subtask]:
    # For this MVP: they are already in good order; dependencies are simple (prev code → tests)
    return subtasks

def extract_text(model_result: Dict) -> str:
    """
    Be robust to different adapter shapes:
    - our adapter often returns {"echo": "...", "tokens": ...}
    - other wrappers might return {"response": "..."} or {"text": "..."} or {"message":{"content": "..."}}
    """
    if not isinstance(model_result, dict):
        return str(model_result)
    return (
        model_result.get("echo")
        or model_result.get("response")
        or model_result.get("text")
        or (model_result.get("message") or {}).get("content")
        or model_result.get("output")
        or ""
    )

# ── API ───────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "router": ROUTER_URL,
        "step_tags": ROUTER_STEP_TAGS,
        "run_id": RUN_ID,
        "dataset_tag": DATASET_TAG,
    }

@app.post("/plan", response_model=PlanOut)
def make_plan(inp: PlanIn):
    sts = plan(inp.prompt)
    return {"subtasks": [st.model_dump() for st in sts]}

@app.post("/plan_run", response_model=RunOut)
async def plan_and_run(inp: PlanIn):
    # 1) plan
    sts = plan(inp.prompt)
    order = topo_order(sts)

    # 2) run in sequence, injecting dependencies
    outputs: Dict[int, str] = {}
    steps: List[Dict] = []
    for st in order:
        prompt_text = st.text
        if st.depends_on is not None and st.depends_on in outputs:
            prompt_text = f"{st.text}\n\nContext from step {st.depends_on}:\n{outputs[st.depends_on]}\n"

        # pass step_id through
        res = await call_router_infer(prompt_text, inp.max_new_tokens or 256, step_tag=st.type, step_id=st.id)
        # router returns: {"backend": i, "backend_url": url, "latency_s": dt, "step_tag": tag, "result": {...}}
        model_text = extract_text(res.get("result"))

        outputs[st.id] = model_text
        steps.append({
            "id": st.id,
            "type": st.type,
            "depends_on": st.depends_on,
            "router_step_tag": res.get("step_tag"),
            "chosen_backend": res.get("backend"),
            "backend_url": res.get("backend_url"),
            "latency_s": res.get("latency_s"),
            "text_in": prompt_text[:4000],
            "text_out": model_text[:4000],
        })

    # 3) simple aggregation: join in order (you can customize per type)
    final = []
    for s in steps:
        header = f"Step {s['id']} ({s['type']} → backend {s.get('chosen_backend')}):"
        final.append(f"{header}\n{s['text_out']}\n")
    final_answer = "\n".join(final).strip()

    return {"steps": steps, "final_answer": final_answer}
