# api_v3.py — tiny wrapper that guarantees `app` exists and exposes a unique /health
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute

# Start with a fallback app so Uvicorn always has something to import
_fallback = FastAPI(title="Elephant Translator — wrapper")
_fallback.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=False
)

try:
    # Try importing the real backend from mini_api.py
    from mini_api import app as _real
    app = _real

    # Remove any existing GET /health so we can override it clearly
    kept = []
    for r in app.router.routes:
        if isinstance(r, APIRoute) and r.path == "/health" and "GET" in r.methods:
            continue
        kept.append(r)
    app.router.routes = kept

    @app.get("/health")
    def health():
        return {"status":"ok","api":"reset-stable v1 (api_v3 wrapper)"}

except Exception as e:
    # mini_api import failed — use fallback app with explanatory health
    app = _fallback

    @app.get("/health")
    def health():
        return {"status":"ok","api":"wrapper-only (mini_api import failed)","error":str(e)}
