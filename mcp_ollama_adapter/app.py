import os, time, json, asyncio
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "mistral:7b-instruct")

app = FastAPI(title=f"MCP Adapter → {MODEL_NAME}")

class InferIn(BaseModel):
    prompt: str
    max_new_tokens: int | None = 256

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "ollama": OLLAMA_BASE_URL}

async def _post_json(client: httpx.AsyncClient, url: str, payload: dict):
    # small retry for transient 5xx or connection errors
    for attempt in range(3):
        try:
            r = await client.post(url, json=payload)
            # retry on 5xx; break for others
            if r.status_code >= 500:
                await asyncio.sleep(0.25 * (attempt + 1))
                continue
            return r
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError):
            await asyncio.sleep(0.25 * (attempt + 1))
    # last try (no swallow)
    return await client.post(url, json=payload)

@app.post("/infer")
async def infer(inp: InferIn):
    t0 = time.perf_counter()
    opts = {"num_predict": int(inp.max_new_tokens or 256)}

    gen_url  = f"{OLLAMA_BASE_URL}/api/generate"
    chat_url = f"{OLLAMA_BASE_URL}/api/chat"

    async with httpx.AsyncClient(timeout=300.0) as client:
        # 1) Try /api/generate
        r = await _post_json(client, gen_url, {
            "model": MODEL_NAME,
            "prompt": inp.prompt,
            "stream": False,
            "options": opts,
        })

        # 2) If generate isn’t supported for this model/format, try /api/chat
        if r.status_code in (404, 405):
            r = await _post_json(client, chat_url, {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": inp.prompt}],
                "stream": False,
                "options": opts,
            })

    if r.status_code != 200:
        # Try to include a short, safe body snippet for debugging
        body = r.text
        snippet = body[:300] if isinstance(body, str) else str(body)[:300]
        raise HTTPException(
            status_code=502,
            detail=f"Ollama {r.status_code} for {MODEL_NAME} at {OLLAMA_BASE_URL}: {snippet}"
        )

    # normalize response across generate/chat shapes
    try:
        j = r.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="Ollama returned non-JSON body")

    text = (
        j.get("response")
        or (j.get("message") or {}).get("content")
        or j.get("output")
        or j.get("text")
        or ""
    )

    return {
        "tokens": max(1, len((text or '').split())),
        "echo": (text or "")[:2048],
        "model": MODEL_NAME,
        "latency_adapter_s": time.perf_counter() - t0
    }
