# mini_api.py — standalone FastAPI with light audio + motion analysis.
# Works even if numpy/Pillow/ffmpeg are missing (degrades gracefully).

import os, io, math, json, shutil, subprocess, tempfile, uuid, logging, wave, contextlib, glob
from pathlib import Path
from typing import List, Dict, Any, Tuple

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Optional deps (graceful fallback if missing)
try:
    import numpy as np
except Exception:
    np = None

try:
    from PIL import Image
except Exception:
    Image = None

log = logging.getLogger("mini_api")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = FastAPI(title="Translator MINI API (enhanced)")

# CORS — allow your GitHub Pages origin (scheme+host only), no credentials.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://cannonballden.github.io"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# Limits & formats
MAX_MB = 25
SUPPORTED_VIDEO = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".ogv"}
SUPPORTED_AUDIO = {".mp3", ".wav", ".ogg", ".oga", ".flac"}

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads_tmp"
FRAME_DIR = BASE_DIR / "frames_tmp"
UPLOAD_DIR.mkdir(exist_ok=True)
FRAME_DIR.mkdir(exist_ok=True)

def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def _run(cmd: List[str]) -> Tuple[bool, str]:
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, text=True)
        return (res.returncode == 0, res.stdout)
    except Exception as e:
        return (False, str(e))

def convert_to_wav(src: Path, dst: Path, sr: int = 16000) -> bool:
    """
    Convert any media to mono 16 kHz WAV with ffmpeg. Returns True on success.
    """
    if not has_ffmpeg():
        return False
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", str(sr), "-vn", str(dst)]
    ok, out = _run(cmd)
    if not ok:
        log.warning("ffmpeg failed: %s", out)
    return ok

def load_wav_pcm16(path: Path) -> Tuple[Any, int]:
    """
    Load mono WAV PCM16 with Python stdlib. Returns (np.float32 array [-1,1], sample_rate).
    If numpy missing, returns (None, sample_rate) and we will degrade.
    """
    with contextlib.closing(wave.open(str(path), "rb")) as wf:
        n_channels = wf.getnchannels()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(n_frames)

    if np is None:
        return None, sr

    if sampwidth == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        # Non-PCM16 → fallback scale
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        x = (x - 128.0) / 128.0

    if n_channels > 1:
        x = x.reshape(-1, n_channels).mean(axis=1)
    return x, sr

def frame_audio(x: "np.ndarray", sr: int, win_s: float = 1.0, hop_s: float = 0.5) -> List[Tuple[int, int]]:
    win = int(sr * win_s)
    hop = int(sr * hop_s)
    idx = []
    start = 0
    while start + win <= len(x):
        idx.append((start, start + win))
        start += hop
    if not idx and len(x) > 0:
        idx.append((0, len(x)))
    return idx

def spectral_features(frame: "np.ndarray", sr: int) -> Dict[str, float]:
    # Hamming window
    w = np.hamming(len(frame))
    f = np.fft.rfft(frame * w)
    mag = np.abs(f) + 1e-9
    freqs = np.fft.rfftfreq(len(frame), 1.0 / sr)

    total = mag.sum()
    centroid = float((freqs * mag).sum() / total)

    low = float(mag[(freqs < 80)].sum())
    mid = float(mag[(freqs >= 100) & (freqs < 300)].sum())
    high = float(mag[(freqs >= 1000) & (freqs < 4000)].sum())
    low_r = low / total
    mid_r = mid / total
    high_r = high / total

    # naive energy
    rms = float(np.sqrt(np.mean(frame ** 2)))

    return {
        "rms": rms,
        "centroid": centroid,
        "low_r": low_r,
        "mid_r": mid_r,
        "high_r": high_r,
    }

def classify_audio_frames(x: "np.ndarray", sr: int) -> Tuple[List[Dict[str, Any]], float]:
    idx = frame_audio(x, sr, win_s=1.0, hop_s=0.5)
    feats = [spectral_features(x[s:e], sr) for (s, e) in idx]
    rms_vals = np.array([f["rms"] for f in feats], dtype=np.float32)
    if len(rms_vals) == 0:
        return [], 0.0
    energy_med = float(np.median(rms_vals))
    energy_hi = float(np.quantile(rms_vals, 0.75))
    dur = len(x) / sr

    labels = []
    t0 = 0.0
    for (i, (s, e)) in enumerate(idx):
        f = feats[i]
        start_t = s / sr
        end_t = e / sr
        label = "calling"
        reason = []
        conf = 0.55

        # Rules (very simple but practical):
        # trumpet: high energy + high high-band and centroid > 1200 Hz
        if f["rms"] > max(energy_hi, energy_med * 1.6) and f["high_r"] > 0.35 and f["centroid"] > 1200:
            label = "trumpet"
            # confidence ~ blend of features
            conf = min(0.95, 0.6 + 0.4 * (f["high_r"] + (f["centroid"] - 1200) / 2000))
            reason.append("high-band energy & high centroid")

        # rumble: energy > median and strong <80 Hz with low centroid
        elif f["rms"] > energy_med * 0.8 and f["low_r"] > 0.30 and f["centroid"] < 200:
            label = "rumble"
            conf = min(0.9, 0.55 + 0.5 * (f["low_r"]))
            reason.append("strong low-frequency energy")

        # growl: mid-band 100–300 Hz prominent, centroid under ~400 Hz
        elif f["mid_r"] > 0.25 and f["centroid"] < 400 and f["rms"] > energy_med * 0.9:
            label = "growl"
            conf = min(0.85, 0.5 + 0.5 * f["mid_r"])
            reason.append("mid-band energy peak")

        # resting: very low energy
        elif f["rms"] < energy_med * 0.6:
            label = "resting"
            conf = min(0.8, 0.6 + (energy_med * 0.6 - f["rms"]) * 5.0)
            reason.append("low energy")

        else:
            label = "calling"
            conf = 0.55
            reason.append("tonal/ambiguous")

        labels.append({
            "start": float(start_t),
            "end": float(end_t),
            "label": label,
            "explanation": f"rms={f['rms']:.3f}, centroid={f['centroid']:.0f} Hz, "
                           f"bands(low={f['low_r']:.2f}, mid={f['mid_r']:.2f}, high={f['high_r']:.2f}); "
                           f"reason: {', '.join(reason)}",
            "confidence": float(max(0.0, min(conf, 0.99))),
            "features": f,
        })

    # Merge contiguous same-label frames
    merged: List[Dict[str, Any]] = []
    for seg in labels:
        if merged and merged[-1]["label"] == seg["label"] and abs(merged[-1]["end"] - seg["start"]) < 1e-6:
            merged[-1]["end"] = seg["end"]
            merged[-1]["confidence"] = float((merged[-1]["confidence"] + seg["confidence"]) / 2.0)
        else:
            merged.append(seg)
    return merged, float(dur)

def extract_frames(path: Path, out_dir: Path, fps: float = 1.0, width: int = 224) -> bool:
    """
    Extract grayscale frames with ffmpeg. Returns True if succeeded.
    """
    if not has_ffmpeg():
        return False
    # Clear target dir
    for f in glob.glob(str(out_dir / "frame_*.png")):
        try:
            os.remove(f)
        except Exception:
            pass
    cmd = [
        "ffmpeg", "-y", "-i", str(path),
        "-vf", f"fps={fps},scale={width}:-1,format=gray",
        str(out_dir / "frame_%05d.png")
    ]
    ok, out = _run(cmd)
    if not ok:
        log.warning("ffmpeg frames failed: %s", out)
    return ok

def movement_from_frames(out_dir: Path) -> Dict[int, float]:
    """
    Compute per-second movement index from consecutive frame diffs (mean absolute diff).
    Requires Pillow + numpy.
    Returns dict {second_index: movement_value}.
    """
    if Image is None or np is None:
        return {}
    files = sorted(Path(out_dir).glob("frame_*.png"))
    if len(files) < 2:
        return {}
    vals = []
    for a, b in zip(files[:-1], files[1:]):
        ia = np.array(Image.open(a), dtype=np.float32)
        ib = np.array(Image.open(b), dtype=np.float32)
        diff = np.mean(np.abs(ia - ib)) / 255.0  # 0..1
        vals.append(float(diff))
    mv = {i: v for i, v in enumerate(vals)}
    return mv

def movement_level(v: float, med: float, hi: float) -> Tuple[str, float]:
    if v >= hi:
        return "high", 0.9
    if v >= med:
        return "medium", 0.7
    return "low", 0.6

def fuse_audio_motion(segments: List[Dict[str, Any]], motion: Dict[int, float]) -> List[Dict[str, Any]]:
    if not segments:
        return segments
    if not motion:
        # No motion info; keep audio-only
        for s in segments:
            s["movement"] = {"index": None, "level": "unknown"}
        return segments

    vals = list(motion.values())
    if not vals:
        for s in segments:
            s["movement"] = {"index": None, "level": "unknown"}
        return segments
    med = float(np.median(vals)) if np is not None else sum(vals)/max(1,len(vals))
    hi = sorted(vals)[int(max(0, len(vals)*0.75))-1] if len(vals) >= 2 else med

    for s in segments:
        # Approximate: pick central second of the segment to sample motion
        sec_idx = int((s["start"] + s["end"]) / 2.0)
        mv_val = motion.get(sec_idx, med)
        level, mconf = movement_level(mv_val, med, hi)
        s["movement"] = {"index": float(mv_val), "level": level}
        # simple fusion: 0.7*audio + 0.3*motion clarity
        s["confidence"] = float(min(0.99, 0.7 * s["confidence"] + 0.3 * mconf))
        # Behavioral phrasing
        if s["label"] == "trumpet" and level == "high":
            s["label"] = "alarm/excitement (trumpet)"
        elif s["label"] == "rumble" and level in ("low", "medium"):
            s["label"] = "contact/coordination (rumble)"
        elif s["label"] == "growl":
            s["label"] = "threat/defensive (growl)"
    return segments

def summarize(segments: List[Dict[str, Any]]) -> Tuple[str, float]:
    if not segments:
        return ("Insufficient evidence to interpret.", 0.4)
    # Pick top label by total duration
    spans: Dict[str, float] = {}
    for s in segments:
        spans[s["label"]] = spans.get(s["label"], 0.0) + (s["end"] - s["start"])
    top = max(spans.items(), key=lambda kv: kv[1])[0]
    conf = float(np.median([s["confidence"] for s in segments])) if np is not None else 0.7
    # Build compact, evidence-based sentence
    if "trumpet" in top:
        msg = "Interpretation: alarm/excitement. Evidence: bright trumpet-like energy; consider giving space."
    elif "rumble" in top:
        msg = "Interpretation: contact/coordination rumbles. Evidence: low-frequency energy; calm or routine context."
    elif "growl" in top:
        msg = "Interpretation: defensive/threat. Evidence: mid-band growl energy; keep distance."
    elif "resting" in top:
        msg = "Interpretation: resting/low arousal. Evidence: sustained low energy."
    else:
        msg = "Interpretation: general calling/ambiguous. Evidence: tonal activity without strong cues."
    return (msg, conf)

def analyze_media_file(path: Path) -> Dict[str, Any]:
    """
    Full pipeline: decode audio, simple audio classification, optional motion from frames.
    """
    diagnostics = {
        "ffmpeg": has_ffmpeg(),
        "numpy": bool(np is not None),
        "pillow": bool(Image is not None),
    }

    # Try to get WAV for audio analysis
    wav_tmp = UPLOAD_DIR / f"conv_{uuid.uuid4().hex}.wav"
    audio_ok = False
    motion_info: Dict[int, float] = {}

    try:
        if diagnostics["ffmpeg"]:
            audio_ok = convert_to_wav(path, wav_tmp)
        else:
            # If already wav, accept; else cannot decode
            if path.suffix.lower() == ".wav":
                wav_tmp = path
                audio_ok = True

        segments: List[Dict[str, Any]] = []
        duration = 0.0

        if audio_ok and np is not None:
            x, sr = load_wav_pcm16(wav_tmp)
            if x is not None:
                segments, duration = classify_audio_frames(x, sr)
        elif audio_ok:
            # WAV read but numpy missing → skip features
            with contextlib.closing(wave.open(str(wav_tmp), "rb")) as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                duration = n_frames / max(1, sr)
            segments = [{
                "start": 0.0, "end": float(duration), "label": "calling",
                "explanation": "Basic decode without numpy; detailed features unavailable.",
                "confidence": 0.6, "features": {}
            }]
        else:
            # Could not decode audio
            segments = [{
                "start": 0.0, "end": 5.0, "label": "uncertain",
                "explanation": "Could not decode audio (install ffmpeg or upload WAV).",
                "confidence": 0.45, "features": {}
            }]
            duration = 5.0

        # Optional: motion from frames for video
        if diagnostics["ffmpeg"]:
            frames_ok = extract_frames(path, FRAME_DIR, fps=1.0, width=224)
            if frames_ok and Image is not None and np is not None:
                motion_info = movement_from_frames(FRAME_DIR)

        segments = fuse_audio_motion(segments, motion_info)
        summary_text, overall_conf = summarize(segments)

        return {
            "file": path.name,
            "species": {"label": "African elephant (heuristic)", "confidence": 0.8},
            "segments": segments,
            "summary": summary_text,
            "overall_confidence": float(min(0.99, overall_conf)),
            "diagnostics": diagnostics,
            "duration": duration
        }
    finally:
        # Cleanup
        if wav_tmp != path:
            try:
                wav_tmp.unlink(missing_ok=True)
            except Exception:
                pass
        # Remove extracted frames
        for f in glob.glob(str(FRAME_DIR / "frame_*.png")):
            try:
                os.remove(f)
            except Exception:
                pass

@app.get("/health")
def health():
    return {"status": "ok", "api": "mini+features"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_VIDEO.union(SUPPORTED_AUDIO):
        return JSONResponse(status_code=400, content={"error": f"Unsupported type {ext}"})

    # Save stream with server-side size limit
    tmp_path = UPLOAD_DIR / f"up_{uuid.uuid4().hex}{ext}"
    size = 0
    with tmp_path.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_MB * 1024 * 1024:
                out.close()
                tmp_path.unlink(missing_ok=True)
                return JSONResponse(status_code=413, content={"error": f"File too large (> {MAX_MB} MB)"})
            out.write(chunk)

    try:
        result = analyze_media_file(tmp_path)
        return JSONResponse(content=result)
    except Exception as e:
        log.exception("Analysis failed")
        return JSONResponse(status_code=500, content={"error": f"Analysis failed: {e}"})
    finally:
        tmp_path.unlink(missing_ok=True)
