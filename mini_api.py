# mini_api.py  — standalone, no imports from your repo

import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="Translator MINI API")

# CORS: your GitHub Pages origin only; no credentials
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://cannonballden.github.io"],  # scheme+host only
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads_tmp"
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_MB = 25
SUPPORTED_VIDEO = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".ogv"}
SUPPORTED_AUDIO = {".mp3", ".wav", ".ogg", ".oga", ".flac"}

@app.get("/health")
def health():
    return {"status": "ok", "api": "mini"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Standalone: no repo imports; always returns a safe stub
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_VIDEO.union(SUPPORTED_AUDIO):
        return JSONResponse(status_code=400, content={"error": f"Unsupported type {ext}"})

    tmp_name = f"{uuid.uuid4().hex}{ext}"
    dst = UPLOAD_DIR / tmp_name

    size = 0
    with dst.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_MB * 1024 * 1024:
                out.close()
                dst.unlink(missing_ok=True)
                return JSONResponse(status_code=413, content={"error": f"File too large (> {MAX_MB} MB)"})
            out.write(chunk)

    # Minimal, deterministic stub result
    result = {
        "file": file.filename,
        "species": {"label": "African elephant (stub)", "confidence": 0.82},
        "segments": [
            {"start": 0.0, "end": 2.5, "label": "calling",
             "explanation": "Tonal energy present (stub).", "confidence": 0.76,
             "features": {"audio_bands": [400, 600], "motion": "low"}},
            {"start": 2.5, "end": 6.0, "label": "resting",
             "explanation": "Low motion; no high-energy vocalization (stub).", "confidence": 0.71,
             "features": {"audio_bands": [], "motion": "very_low"}}
        ],
        "summary": "Likely routine contact call; no threat signals (stub).",
        "overall_confidence": 0.74
    }
    dst.unlink(missing_ok=True)
    return JSONResponse(content=result)
