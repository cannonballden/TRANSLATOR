# api_v3.py — safe wrapper that never 500s on /health
# Behavior:
# - If mini_api imports: expose its FastAPI app (so /health returns the mini_api version string).
# - If import fails: expose a tiny fallback FastAPI app with a helpful /health.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Try to import the real app
IMPORT_OK = False
IMPORT_ERR = ""
core_app = None
core_health = None

try:
    import mini_api  # this should define 'app' and 'health'
    core_app = mini_api.app
    core_health = getattr(mini_api, "health", None)
    IMPORT_OK = True
except Exception as err:  # never reference 'e' out of scope again
    IMPORT_OK = False
    IMPORT_ERR = repr(err)
    core_app = None
    core_health = None

if IMPORT_OK and core_app is not None:
    # Use the real app directly so routes (including /health) are identical.
    app = core_app

    # Add an extra diagnostic endpoint without touching core routes.
    @app.get("/wrapper-health")
    def wrapper_health():
        info = {"import_ok": True, "wrapper": "ok"}
        try:
            if callable(core_health):
                info["core_health"] = core_health()
        except Exception as inner_err:
            info["core_health_error"] = repr(inner_err)
        return info

else:
    # Fallback app that always answers /health with the import error.
    app = FastAPI(title="Translator API wrapper (fallback)")

    # Keep CORS permissive for the demo (GH Pages -> Codespaces)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "api": "wrapper-only (mini_api import failed)",
            "error": IMPORT_ERR
        }

    @app.get("/wrapper-health")
    def wrapper_health():
        return {
            "import_ok": False,
            "import_error": IMPORT_ERR
        }
