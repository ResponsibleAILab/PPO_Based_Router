# =============================
# File: mcp_sim/app.py
# =============================
import os, random, time
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="MCP Sim Backend")

class InferIn(BaseModel):
    prompt: str
    max_new_tokens: int | None = 64

INFER_MEAN_MS = int(os.getenv("INFER_MEAN_MS", "120"))
INFER_JITTER_MS = int(os.getenv("INFER_JITTER_MS", "60"))
FAIL_RATE = float(os.getenv("FAIL_RATE", "0.0"))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/infer")
def infer(inp: InferIn):
    if random.random() < FAIL_RATE:
        return {"error": "simulated failure"}
    jitter = random.randint(0, INFER_JITTER_MS)
    delay_ms = max(1, int(random.gauss(INFER_MEAN_MS, INFER_JITTER_MS/2))) + jitter
    time.sleep(delay_ms / 1000.0)
    return {"tokens": min((len(inp.prompt)//12)+1, inp.max_new_tokens or 64), "echo": inp.prompt[:128]}