
# server.py
# FastAPI backend for a sleek, streaming chat UI that talks to Ollama.
# Usage:
#   1) pip install -r requirements.txt
#   2) bash start.sh
#   3) http://localhost:8000

import os
import json
import math
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure via env if you want
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("MODEL", "goekdenizguelmez/JOSIEFIED-Qwen3:4b-q5_k_m")
DEFAULT_NUM_CTX = int(os.getenv("DEFAULT_NUM_CTX", "8192"))
USER_MAX_CTX = int(os.getenv("USER_MAX_CTX", "8192"))
CTX_MULTIPLIER = float(os.getenv("CTX_MULTIPLIER", "1.1"))

app = FastAPI(title="Ollama Chat UI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def estimate_tokens(text: str) -> int:
    # Rough heuristic: ~4 chars/token
    return math.ceil(len(text) / 4) if text else 0


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def build_options(settings: Dict[str, Any], messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    temperature = float(settings.get("temperature", 0.9))
    top_p = float(settings.get("top_p", 0.9))
    top_k = int(settings.get("top_k", 100))
    seed = settings.get("seed") or None
    num_predict = settings.get("num_predict") or None

    dynamic_ctx = bool(settings.get("dynamic_ctx", False))
    user_max_ctx = int(settings.get("max_ctx", USER_MAX_CTX))
    static_ctx = int(settings.get("num_ctx", DEFAULT_NUM_CTX))

    joined = "".join(f"{m.get('role','')}: {m.get('content','')}\n" for m in messages)
    est_tokens = estimate_tokens(joined)

    if dynamic_ctx:
        num_ctx = clamp(int(est_tokens * CTX_MULTIPLIER), 4096, user_max_ctx)
    else:
        num_ctx = clamp(static_ctx, 2048, user_max_ctx)

    opts: Dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "num_ctx": num_ctx,
    }
    if seed not in (None, ""):
        try: opts["seed"] = int(seed)
        except: pass
    if num_predict not in (None, ""):
        try: opts["num_predict"] = int(num_predict)
        except: pass

    return opts


@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(os.path.join(APP_DIR, "index.html"))


@app.get("/api/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags")
            ok = (r.status_code == 200)
    except Exception:
        ok = False
    return {"ok": ok, "ollama": OLLAMA_HOST, "model": MODEL}


@app.post("/api/chat/stream")
async def chat_stream(payload: Dict[str, Any]):
    messages = payload.get("messages", [])
    settings = payload.get("settings", {})
    system_prompt = payload.get("system") or None

    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages

    options = build_options(settings, messages)

    async def event_gen():
        req = {"model": MODEL, "messages": messages, "stream": True, "options": options}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", f"{OLLAMA_HOST}/api/chat", json=req) as resp:
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except Exception:
                            continue

                        msg = data.get("message", {})
                        if isinstance(msg, dict) and "content" in msg:
                            yield f"data: {json.dumps({'type':'delta','delta': msg['content']})}\n\n"

                        if "tool_calls" in data:
                            yield f"data: {json.dumps({'type':'tool_calls','tool_calls': data['tool_calls']})}\n\n"

                        if data.get("done"):
                            metrics = data.get("metrics", {})
                            usage = {
                                "prompt_eval_count": metrics.get("prompt_eval_count"),
                                "eval_count": metrics.get("eval_count"),
                                "total_duration_ms": int(metrics.get("total_duration", 0) / 1e6) if metrics.get("total_duration") else None,
                                "eval_duration_ms": int(metrics.get("eval_duration", 0) / 1e6) if metrics.get("eval_duration") else None,
                            }
                            yield f"data: {json.dumps({'type':'done','options': options, 'usage': usage})}\n\n"
                            break
        except httpx.RequestError as e:
            yield f"data: {json.dumps({'type':'error','message': f'Backend request failed: {e}'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','message': f'Unexpected error: {e.__class__.__name__}: {e}'})}\n\n"

        yield "event: close\ndata: {}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/api/models/set")
async def set_model(payload: Dict[str, Any]):
    global MODEL
    MODEL = payload.get("model", MODEL)
    return {"ok": True, "model": MODEL}


@app.get("/api/models")
async def list_models():
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{OLLAMA_HOST}/api/tags")
        return JSONResponse(r.json())
