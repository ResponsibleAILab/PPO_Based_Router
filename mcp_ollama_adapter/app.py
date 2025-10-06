import os, time, httpx
from fastapi import FastAPI
from pydantic import BaseModel

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "mistral:7b-instruct")

app = FastAPI(title=f"MCP Adapter → {MODEL_NAME}")

class InferIn(BaseModel):
    prompt: str
    max_new_tokens: int | None = 256

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/infer")
async def infer(inp: InferIn):
    t0 = time.perf_counter()
    payload = {
        "model": MODEL_NAME,
        "prompt": inp.prompt,
        "stream": False,
        "options": {"num_predict": inp.max_new_tokens or 256}
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        r.raise_for_status()
        j = r.json()
    # Return in the same simple shape your router expects
    text = j.get("response", "")
    tokens_out = max(1, len(text.split()))
    return {"tokens": tokens_out, "echo": text[:512], "model": MODEL_NAME, "latency_adapter_s": time.perf_counter()-t0}
