# api_v3.py — unified wrapper that surfaces inner mini_api health/config/analyze
import reprlib
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

ALLOW_ORIGINS = ["*"]
app = FastAPI(title="Translator API wrapper v3 (unified health)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

try:
    import mini_api as inner
    INNER_OK = True
    INNER_ERR = ""
except Exception as e:
    inner = None
    INNER_OK = False
    INNER_ERR = repr(e)

@app.get("/health")
def health():
    if INNER_OK:
        try:
            inner_h = inner.health()
            return {"status":"ok", "api":"wrapper v3 + inner", "inner": inner_h}
        except Exception as e:
            return {"status":"ok", "api":"wrapper v3 (inner call failed)", "error": repr(e)}
    else:
        return {"status":"ok", "api":"wrapper-only (mini_api import failed)", "error": INNER_ERR}

@app.get("/config")
def config():
    if INNER_OK:
        try:
            return inner.config()
        except Exception as e:
            return {"error": f"inner config failed: {e}"}
    return {"error": "mini_api not available"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), include_motion: bool = True):
    if INNER_OK:
        try:
            return await inner.analyze(file=file, include_motion=include_motion)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"inner analyze failed: {e}"})
    return JSONResponse(status_code=500, content={"error":"mini_api not available"})
