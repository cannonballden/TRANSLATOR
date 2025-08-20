# api_v3.py — safe wrapper that never mutates routes
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Fallback app, so uvicorn always has an ASGI app to import
_fallback = FastAPI(title="Elephant Translator — wrapper")
_fallback.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=False
)

try:
    # Import the real app and just ADD an extra endpoint (no overriding)
    from mini_api import app as real_app
    app = real_app

    @app.get("/wrapper-health")
    def wrapper_health():
        return {"status": "ok", "api": "reset-stable v1 (wrapper alive)"}

except Exception as e:
    # If import fails, expose a minimal app that explains why
    app = _fallback

    @app.get("/health")
    def health():
        return {"status":"ok","api":"wrapper-only (mini_api import failed)","error":str(e)}
