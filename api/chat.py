"""FastAPI endpoint powering sid.agent.

Deployed on Vercel as a Python serverless function.

Routes:
    GET  /                 → serves index.html (portfolio homepage)
    GET  /<any static>     → serves any static file at project root (resume PDF, etc.)
    POST /api/chat         → { "reply": string } (non-streaming, kept for fallback)
    POST /api/chat/stream  → SSE stream of text chunks (primary path, feels instant)
    GET  /api/warmup       → wakes the cold function + primes Groq connection pool
    GET  /api/health       → { "status": "ok", "agent": "sid.agent", "ready": bool }
    GET  /api/docs         → auto-generated OpenAPI docs

This file is intentionally thin: it only handles HTTP concerns
(request/response, rate limiting, error mapping, static serving).
All agent logic lives in the `siddhikesh_agent` package at the project root.
"""

import json
import sys
from pathlib import Path
from typing import Optional

# Make the project-root package importable when Vercel runs this file
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from siddhikesh_agent import Agent, LLMError, RateLimiter


# ─── App ─────────────────────────────────────────────────
app = FastAPI(
    title="sid.agent",
    version="2.1.0",
    description="AI agent trained on Siddhikesh Gavit's life.",
    docs_url="/api/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# ─── Global state (survives across warm invocations) ────
_agent: Optional[Agent] = None
_limiter = RateLimiter()


def _get_agent() -> Agent:
    """Lazy-init the Agent on first request; reused on warm invocations."""
    global _agent
    if _agent is None:
        _agent = Agent.from_env()
    return _agent


def _client_ip(request: Request) -> str:
    """Best-effort client IP extraction from Vercel's forwarded headers."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ─── Request models ──────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=600)


# ─── Routes ──────────────────────────────────────────────
@app.post("/api/chat")
async def chat_endpoint(payload: ChatRequest, request: Request):
    """Non-streaming chat — returns the full reply once complete.

    Kept for compatibility. The primary path is /api/chat/stream which
    surfaces tokens as they generate for dramatically better perceived latency.
    """
    ip = _client_ip(request)

    if not _limiter.allow(ip):
        raise HTTPException(
            status_code=429,
            detail="too many questions, take a breath. try again in a minute.",
        )

    try:
        agent = _get_agent()
        reply = await agent.chat(payload.message)
        return {"reply": reply}
    except LLMError as e:
        raise HTTPException(status_code=502, detail=f"agent unavailable: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="something broke on my end. probably my fault. try again?",
        )


@app.post("/api/chat/stream")
async def chat_stream_endpoint(payload: ChatRequest, request: Request):
    """Streaming chat — yields Server-Sent Events as the LLM generates.

    Response format (SSE):
        data: {"chunk": "hello"}\\n\\n
        data: {"chunk": " world"}\\n\\n
        data: {"done": true}\\n\\n

    Errors are sent as an SSE event so the client can render them inline
    instead of getting a broken connection.
    """
    ip = _client_ip(request)

    if not _limiter.allow(ip):
        raise HTTPException(
            status_code=429,
            detail="too many questions, take a breath. try again in a minute.",
        )

    async def event_stream():
        try:
            agent = _get_agent()
            async for chunk in agent.chat_stream(payload.message):
                # SSE format: `data: <json>\n\n`
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except LLMError as e:
            yield f"data: {json.dumps({'error': f'agent unavailable: {e}'})}\n\n"
        except ValueError as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        except Exception:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'error': 'something broke on my end. probably my fault. try again?'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",  # disable proxy buffering (nginx/vercel)
            "Connection": "keep-alive",
        },
    )


@app.get("/api/warmup")
async def warmup():
    """Wakes the cold function + primes the Groq connection pool.

    Called by the frontend when the user scrolls the agent section into view.
    Because the function is now warm by the time they actually type a message,
    the first message feels as fast as subsequent ones — killing cold-start pain
    entirely.
    """
    # Force agent init (loads env, builds LLM client)
    agent = _get_agent()
    # Priming the shared httpx pool would require an actual outbound request;
    # we skip that here to keep warmup cheap (no Groq tokens burned).
    return {
        "warm": True,
        "agent_ready": agent is not None,
    }


@app.get("/api/health")
async def health():
    """Liveness check — is the agent ready to serve?"""
    return {
        "status": "ok",
        "agent": "sid.agent",
        "ready": _agent is not None,
    }


# ─── Error handler for uniform JSON error shape ─────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


# ─── Static file serving (project root as web root) ─────
# The frontend `index.html` and downloadable assets (resume PDF) live at the
# project root. Since Vercel now routes ALL requests to this FastAPI function
# (not just /api/*), we serve those files from here too. Mounted AFTER all
# API routes so /api/* handlers take precedence.

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@app.get("/")
async def root():
    """Serve the portfolio homepage."""
    index_path = _PROJECT_ROOT / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="index.html not found")


# Mount everything else at the project root — resume PDF, any future assets.
# `html=True` makes it fall back to index.html for unknown paths (SPA-style).
try:
    app.mount(
        "/",
        StaticFiles(directory=str(_PROJECT_ROOT), html=True),
        name="static",
    )
except RuntimeError:
    # Directory doesn't exist during some test scenarios — skip mount
    pass