import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Temporary upload folder
UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Elephant Translator")

# CORS: allow your GitHub Pages origin ONLY (no credentials)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://cannonballden.github.io"],  # origin = scheme+host only
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False
)

# Supported file types and a conservative size limit
SUPPORTED_VIDEO = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".ogv"}
SUPPORTED_AUDIO = {".mp3", ".wav", ".ogg", ".oga", ".flac"}
MAX_MB = 25

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Lazy import so startup is instant
    from .analyzer import analyze_media

    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_VIDEO.union(SUPPORTED_AUDIO):
        return JSONResponse(status_code=400, content={"error": f"Unsupported type {ext}"})

    tmp_name = f"{uuid.uuid4().hex}{ext}"
    dst = UPLOAD_DIR / tmp_name

    # Stream upload to disk with size guard
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

    try:
        result = analyze_media(str(dst))
        return JSONResponse(content=result)
    finally:
        dst.unlink(missing_ok=True)
