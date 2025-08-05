
# server.py
# FastAPI backend for a sleek, streaming chat UI that talks to Ollama.
# Usage:
#   1) pip install -r requirements.txt
#   2) bash start.sh
#   3) http://localhost:8000

import os
import math
from typing import Any, Dict, List

import httpx
import orjson
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from tools import web_search, open_url

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure via env if you want
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("MODEL", "goekdenizguelmez/JOSIEFIED-Qwen3:4b-q5_k_m")
DEFAULT_NUM_CTX = int(os.getenv("DEFAULT_NUM_CTX", "8192"))
USER_MAX_CTX = int(os.getenv("USER_MAX_CTX", "40000"))

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for recent or factual info and return top results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "k": {
                        "type": "integer",
                        "description": "Number of results",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": "Open a URL and return a concise text extract for summarization.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"},
                    "max_chars": {
                        "type": "integer",
                        "description": "Max characters of extracted text",
                        "default": 6000,
                    },
                },
                "required": ["url"],
            },
        },
    },
]

app = FastAPI(title="Ollama Chat UI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    limits = httpx.Limits(max_connections=20, max_keepalive_connections=20)
    app.state.client = httpx.AsyncClient(http2=True, limits=limits)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await app.state.client.aclose()


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

    dynamic_ctx = bool(settings.get("dynamic_ctx", True))
    user_max_ctx = int(settings.get("max_ctx", USER_MAX_CTX))
    static_ctx = int(settings.get("num_ctx", DEFAULT_NUM_CTX))
    num_thread = settings.get("num_thread") or None
    num_batch = settings.get("num_batch") or None
    num_gpu = settings.get("num_gpu") or None

    joined = "".join(f"{m.get('role','')}: {m.get('content','')}\n" for m in messages)
    est_tokens = estimate_tokens(joined)

    if dynamic_ctx:
        num_ctx = clamp(int(est_tokens * 1.2), 4096, user_max_ctx)
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
    if num_thread not in (None, ""):
        try: opts["num_thread"] = int(num_thread)
        except: pass
    if num_batch not in (None, ""):
        try: opts["num_batch"] = int(num_batch)
        except: pass
    if num_gpu not in (None, ""):
        try: opts["num_gpu"] = int(num_gpu)
        except: pass

    return opts


@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(os.path.join(APP_DIR, "index.html"))


@app.get("/api/health")
async def health():
    try:
        r = await app.state.client.get(f"{OLLAMA_HOST}/api/tags", timeout=3.0)
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

    async def event_gen():
        DATA = b"data: "
        END = b"\n\n"
        convo = list(messages)
        client = app.state.client

        while True:
            options = build_options(settings, convo)
            req = {
                "model": MODEL,
                "messages": convo,
                "stream": True,
                "options": options,
                "tools": TOOLS,
            }
            tool_calls = []
            done_payload = None
            try:
                async with client.stream("POST", f"{OLLAMA_HOST}/api/chat", json=req, timeout=None) as resp:
                    buffer = b""
                    async for chunk in resp.aiter_bytes():
                        if not chunk:
                            continue
                        buffer += chunk
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                data = orjson.loads(line)
                            except Exception:
                                continue

                            msg = data.get("message", {})
                            if isinstance(msg, dict) and "content" in msg:
                                yield DATA + orjson.dumps({"type": "delta", "delta": msg["content"]}) + END

                            if "tool_calls" in data:
                                tool_calls = data["tool_calls"]
                                yield DATA + orjson.dumps({"type": "tool_calls", "tool_calls": tool_calls}) + END

                            if data.get("done"):
                                metrics = data.get("metrics", {})
                                usage = {
                                    "prompt_eval_count": metrics.get("prompt_eval_count"),
                                    "eval_count": metrics.get("eval_count"),
                                    "total_duration_ms": int(metrics.get("total_duration", 0) / 1e6) if metrics.get("total_duration") else None,
                                    "eval_duration_ms": int(metrics.get("eval_duration", 0) / 1e6) if metrics.get("eval_duration") else None,
                                }
                                done_payload = {"type": "done", "options": options, "usage": usage}
                                break
            except httpx.RequestError as e:
                yield DATA + orjson.dumps({"type": "error", "message": f"Backend request failed: {e}"}) + END
                break
            except Exception as e:
                yield DATA + orjson.dumps({"type": "error", "message": f"Unexpected error: {e.__class__.__name__}: {e}"}) + END
                break

            if tool_calls:
                convo.append({"role": "assistant", "tool_calls": tool_calls})
                for tc in tool_calls:
                    call_id = tc.get("id")
                    name = tc.get("function", {}).get("name")
                    args = tc.get("function", {}).get("arguments") or {}
                    try:
                        if name == "web_search":
                            payload = web_search(args.get("query", ""), int(args.get("k", 5)))
                        elif name == "open_url":
                            payload = open_url(args["url"], int(args.get("max_chars", 6000)))
                        else:
                            payload = {"error": f"Unknown tool {name}"}
                    except Exception as e:
                        payload = {"error": f"{type(e).__name__}: {e}"}
                    convo.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": orjson.dumps(payload, ensure_ascii=False).decode(),
                    })
                continue

            if done_payload:
                yield DATA + orjson.dumps(done_payload) + END
            break

        yield b"event: close\ndata: {}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/api/models/set")
async def set_model(payload: Dict[str, Any]):
    global MODEL
    MODEL = payload.get("model", MODEL)
    return {"ok": True, "model": MODEL}


@app.get("/api/models")
async def list_models():
    r = await app.state.client.get(f"{OLLAMA_HOST}/api/tags", timeout=10.0)
    return JSONResponse(r.json())
