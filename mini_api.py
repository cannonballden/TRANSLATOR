# mini_api.py — enhanced FastAPI backend for Elephant Translator demo
# - Heuristic audio analysis (band energies, centroid, flatness, rolloff, flux, tonality)
# - Simple video motion cues via frame diffs (ffmpeg + Pillow)
# - Robust fallbacks if numpy/Pillow/ffmpeg are missing
# - Conservative interpretations with clear uncertainty

import os, io, math, json, shutil, subprocess, tempfile, uuid, logging, wave, contextlib, glob
from pathlib import Path
from typing import List, Dict, Any, Tuple

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Optional deps (graceful degradation if missing)
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

app = FastAPI(title="Elephant Translator — MINI API++")

# CORS: your GitHub Pages origin (scheme+host), no credentials
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
    """Convert any media to mono 16 kHz WAV with ffmpeg. Returns True on success."""
    if not has_ffmpeg():
        return False
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", str(sr), "-vn", str(dst)]
    ok, out = _run(cmd)
    if not ok:
        log.warning("ffmpeg failed: %s", out)
    return ok

def load_wav_pcm16(path: Path) -> Tuple[Any, int]:
    """Load mono WAV PCM16 with stdlib 'wave'. Returns (np.float32 array [-1,1], sr). If numpy missing, (None, sr)."""
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
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        x = (x - 128.0) / 128.0

    if n_channels > 1:
        x = x.reshape(-1, n_channels).mean(axis=1)
    return x, sr

def frame_audio(x: "np.ndarray", sr: int, win_s: float = 0.5, hop_s: float = 0.25) -> List[Tuple[int, int]]:
    win = max(1, int(sr * win_s))
    hop = max(1, int(sr * hop_s))
    idx = []
    start = 0
    while start + win <= len(x):
        idx.append((start, start + win))
        start += hop
    if not idx and len(x) > 0:
        idx.append((0, len(x)))
    return idx

def spectral_flatness(mag: "np.ndarray") -> float:
    # geometric mean / arithmetic mean
    eps = 1e-9
    gm = float(np.exp(np.mean(np.log(mag + eps))))
    am = float(np.mean(mag + eps))
    return gm / am

def spectral_rolloff(freqs: "np.ndarray", mag: "np.ndarray", pct: float = 0.85) -> float:
    cumsum = np.cumsum(mag)
    thresh = pct * cumsum[-1]
    idx = np.searchsorted(cumsum, thresh)
    return float(freqs[min(idx, len(freqs)-1)])

def spectral_flux(prev_mag: "np.ndarray", mag: "np.ndarray") -> float:
    if prev_mag is None: return 0.0
    # L2 flux
    d = mag - prev_mag
    return float(np.sqrt(np.sum(d * d)) / (len(d) + 1e-9))

def tonal_peakness(mag: "np.ndarray") -> float:
    # Peak prominence: ratio of top-5 peaks to total energy (rough tonality proxy)
    if len(mag) < 8: return 0.0
    top = np.sort(mag)[-5:].sum()
    return float(top / (mag.sum() + 1e-9))

def spectral_features(frame: "np.ndarray", sr: int, prev_mag: "np.ndarray") -> Tuple[Dict[str, float], "np.ndarray"]:
    w = np.hamming(len(frame))
    f = np.fft.rfft(frame * w)
    mag = np.abs(f) + 1e-9
    freqs = np.fft.rfftfreq(len(frame), 1.0 / sr)

    total = mag.sum()
    centroid = float((freqs * mag).sum() / total)
    flat = spectral_flatness(mag)
    roll85 = spectral_rolloff(freqs, mag, pct=0.85)
    flux = spectral_flux(prev_mag, mag)
    peak_idx = int(np.argmax(mag))
    peak_freq = float(freqs[peak_idx])
    tone = tonal_peakness(mag)

    low = float(mag[(freqs < 80)].sum())
    mid = float(mag[(freqs >= 100) & (freqs < 300)].sum())
    upper_mid = float(mag[(freqs >= 300) & (freqs < 1000)].sum())
    high = float(mag[(freqs >= 1000) & (freqs < 4000)].sum())

    low_r = low / total
    mid_r = mid / total
    upper_mid_r = upper_mid / total
    high_r = high / total

    rms = float(np.sqrt(np.mean(frame ** 2)))

    feat = {
        "rms": rms,
        "centroid": centroid,
        "flatness": flat,
        "roll85": roll85,
        "flux": flux,
        "peak_freq": peak_freq,
        "tonality": tone,
        "low_r": low_r,
        "mid_r": mid_r,
        "upper_mid_r": upper_mid_r,
        "high_r": high_r,
    }
    return feat, mag

def classify_audio(x: "np.ndarray", sr: int) -> Tuple[List[Dict[str, Any]], float]:
    """Return per-frame labels with explanations; merge contiguous segments."""
    idx = frame_audio(x, sr, win_s=0.5, hop_s=0.25)
    feats: List[Dict[str, float]] = []
    prev_mag = None
    for (s, e) in idx:
        f, prev_mag = spectral_features(x[s:e], sr, prev_mag)
        feats.append(f)

    if not feats:
        return [], 0.0

    rms_vals = np.array([f["rms"] for f in feats], dtype=np.float32)
    energy_med = float(np.median(rms_vals))
    energy_hi = float(np.quantile(rms_vals, 0.75))
    dur = len(x) / sr

    # Centroid slope for let's-go rumble (across 3-frame windows)
    cent = np.array([f["centroid"] for f in feats], dtype=np.float32)
    slope = np.zeros_like(cent)
    if len(cent) >= 3:
        slope[1:-1] = (cent[2:] - cent[:-2]) / 2.0

    labels = []
    for i, (samp, (s_s, s_e)) in enumerate(zip(feats, idx)):
        start_t = s_s / sr
        end_t = s_e / sr

        label = "calling"
        conf = 0.55
        reasons = []

        # Heuristic rules grounded in bioacoustic cues
        if samp["rms"] > max(energy_hi, energy_med * 1.6) and samp["high_r"] > 0.35 and samp["centroid"] > 1200:
            label = "trumpet"
            conf = min(0.97, 0.65 + 0.4 * (samp["high_r"]) + 0.0001 * (samp["centroid"] - 1200))
            reasons.append("bright high-band energy & high centroid")

        elif samp["rms"] > energy_med * 0.8 and samp["low_r"] > 0.33 and samp["centroid"] < 180 and samp["flatness"] < 0.5:
            label = "contact rumble"
            conf = min(0.9, 0.6 + 0.5 * samp["low_r"])
            reasons.append("strong low-frequency energy, smooth spectrum")

        elif samp["rms"] > energy_med * 0.9 and samp["low_r"] > 0.28 and slope[i] > 30 and samp["centroid"] < 260:
            label = "let’s‑go rumble (matriarch)"
            conf = min(0.9, 0.55 + 0.5 * min(1.0, slope[i] / 120.0))
            reasons.append("low band + rising centroid/energy")

        elif samp["rms"] > energy_med * 0.9 and samp["low_r"] > 0.25 and samp["flatness"] > 0.6 and samp["centroid"] < 220:
            label = "musth‑like buzz (male)"
            conf = min(0.85, 0.5 + 0.6 * (samp["flatness"] - 0.6 + samp["low_r"]))
            reasons.append("low centroid with buzzy flat spectrum")

        elif samp["rms"] > energy_med and samp["upper_mid_r"] > 0.28 and samp["flatness"] > 0.55:
            label = "roar / rumble‑roar"
            conf = min(0.9, 0.55 + 0.5 * samp["upper_mid_r"])
            reasons.append("harsh mid/upper-mid energy")

        elif 250 <= samp["centroid"] <= 600 and samp["rms"] > energy_med * 0.8 and (s_e - s_s) / sr <= 0.75:
            label = "estrous rumble (female)"
            conf = 0.7
            reasons.append("short, higher‑pitched rumble")

        elif samp["rms"] < energy_med * 0.6:
            label = "resting / low arousal"
            conf = min(0.8, 0.6 + (energy_med * 0.6 - samp["rms"]) * 3.0)
            reasons.append("low overall energy")

        else:
            if samp["tonality"] > 0.22 and samp["low_r"] > 0.2 and 80 < samp["peak_freq"] < 400:
                label = "possible individual‑address rumble (uncertain)"
                conf = 0.6
                reasons.append("tonal narrowband peak")

        labels.append({
            "start": float(start_t),
            "end": float(end_t),
            "label": label,
            "explanation": (
                f"rms={samp['rms']:.3f}, centroid={samp['centroid']:.0f}Hz, flat={samp['flatness']:.2f}, "
                f"bands(low={samp['low_r']:.2f}, mid={samp['mid_r']:.2f}, upperMid={samp['upper_mid_r']:.2f}, high={samp['high_r']:.2f}), "
                f"peak={samp['peak_freq']:.0f}Hz, slope={slope[i]:.1f}; reason: {', '.join(reasons) or 'ambiguous'}"
            ),
            "confidence": float(max(0.0, min(conf, 0.99))),
            "features": samp,
        })

    # Merge contiguous same-label frames
    merged: List[Dict[str, Any]] = []
    for seg in labels:
        if merged and merged[-1]["label"] == seg["label"] and abs(merged[-1]["end"] - seg["start"]) < 1e-6:
            merged[-1]["end"] = seg["end"]
            merged[-1]["confidence"] = float((merged[-1]["confidence"] + seg["confidence"]) / 2.0)
        else:
            merged.append(seg)

    # Greeting chorus detection: if trumpet/roar overlaps with contact/let’s-go in ±2s
    for i in range(len(merged)):
        for j in range(i+1, len(merged)):
            a, b = merged[i], merged[j]
            if b["start"] > a["end"] + 2.0: break
            chorus_pair = (
                ("trumpet" in a["label"] or "roar" in a["label"])
                and ("rumble" in b["label"])
            ) or (
                ("trumpet" in b["label"] or "roar" in b["label"])
                and ("rumble" in a["label"])
            )
            if chorus_pair and max(a["start"], b["start"]) - min(a["end"], b["end"]) < 2.0:
                # Mark both
                a["label"] = "greeting chorus (family reunion)"
                b["label"] = "greeting chorus (family reunion)"
                a["confidence"] = min(0.95, max(a["confidence"], b["confidence"]) + 0.05)
                b["confidence"] = a["confidence"]

    return merged, float(dur)

def extract_frames(path: Path, out_dir: Path, fps: float = 2.0, width: int = 224) -> bool:
    """Extract grayscale frames using ffmpeg. Returns True if succeeded."""
    if not has_ffmpeg():
        return False
    for f in glob.glob(str(out_dir / "frame_*.png")):
        try: os.remove(f)
        except Exception: pass
    cmd = [
        "ffmpeg", "-y", "-i", str(path),
        "-vf", f"fps={fps},scale={width}:-1,format=gray",
        str(out_dir / "frame_%05d.png"),
    ]
    ok, out = _run(cmd)
    if not ok:
        log.warning("ffmpeg frames failed: %s", out)
    return ok

def motion_series(out_dir: Path) -> List[float]:
    """Return motion per step as mean absolute diff between consecutive frames (0..1)."""
    if Image is None or np is None:
        return []
    files = sorted(Path(out_dir).glob("frame_*.png"))
    if len(files) < 2:
        return []
    vals = []
    prev = np.array(Image.open(files[0]), dtype=np.float32)
    for f in files[1:]:
        cur = np.array(Image.open(f), dtype=np.float32)
        vals.append(float(np.mean(np.abs(cur - prev)) / 255.0))
        prev = cur
    return vals  # sampled at ~fps Hz

def detect_motion_cues(vals: List[float], fps: float = 2.0) -> Dict[str, Any]:
    """Return {'series': [...], 'events': [{'t':sec,'type':label,'detail':...}, ...]}"""
    out = {"series": vals, "events": []}
    if not vals:
        return out
    arr = np.array(vals, dtype=np.float32)
    med = float(np.median(arr))
    mean = float(np.mean(arr))
    std = float(np.std(arr))

    # Freeze if ≤ med*0.4 for ≥2 seconds
    low_thresh = med * 0.4
    run = 0
    for i, v in enumerate(arr):
        if v <= low_thresh: run += 1
        else:
            if run >= int(2 * fps):
                t = int((i - run/2) / fps)
                out["events"].append({"t": t, "type": "freeze and listen", "detail": f"very low motion for ~{run/fps:.1f}s"})
            run = 0
    if run >= int(2 * fps):
        t = int((len(arr) - run/2) / fps)
        out["events"].append({"t": t, "type": "freeze and listen", "detail": f"very low motion for ~{run/fps:.1f}s"})

    # Stomp: spike  > med + 2.5*std
    if std > 0:
        spike_idx = np.where(arr > med + 2.5 * std)[0]
        for i in spike_idx:
            t = int(i / fps)
            out["events"].append({"t": t, "type": "foot stomp / seismic", "detail": "sudden motion spike"})

    # Ear flapping periodicity: simple DFT peak
    # Only meaningful if we have ≥8 samples
    if len(arr) >= 8:
        # detrend
        a = arr - np.mean(arr)
        spec = np.abs(np.fft.rfft(a))
        freqs = np.fft.rfftfreq(len(a), d=1.0/fps)
        # Find dominant >0 frequency
        if len(spec) > 1:
            dom_idx = int(np.argmax(spec[1:]) + 1)
            dom_f = float(freqs[dom_idx])
            if 0.2 <= dom_f <= 0.6:
                out["events"].append({"t": None, "type": "ear flapping (calm)", "detail": f"periodicity ~{dom_f:.2f} Hz"})
            elif 0.8 <= dom_f <= 1.5:
                out["events"].append({"t": None, "type": "ear flapping (agitated)", "detail": f"periodicity ~{dom_f:.2f} Hz"})

    # Excitement proxy
    if mean > med * 1.3:
        out["events"].append({"t": None, "type": "high excitement (motion)", "detail": "sustained elevated motion"})
    return out

def fuse_audio_motion(segments: List[Dict[str, Any]], motion: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not segments:
        return segments
    mv_series = motion.get("series", [])
    events = motion.get("events", [])
    if not mv_series:
        for s in segments:
            s["movement"] = {"level": "unknown", "notes": []}
        return segments

    arr = np.array(mv_series, dtype=np.float32)
    med = float(np.median(arr)) if len(arr) else 0.0
    hi = float(np.quantile(arr, 0.75)) if len(arr) else med

    for s in segments:
        # Sample motion at segment midpoint
        sec_idx = int((s["start"] + s["end"]) / 2.0)
        mv_val = arr[min(sec_idx, len(arr)-1)] if len(arr) else med
        level = "low" if mv_val < med*0.8 else ("medium" if mv_val < hi*1.1 else "high")
        notes = []
        # Include notable events near this time
        for ev in events:
            if ev["t"] is not None and abs(ev["t"] - sec_idx) <= 1:
                notes.append(ev["type"])
        s["movement"] = {"level": level, "value": float(mv_val), "notes": notes}
        # Fusion: modest boost if motion supports label meaning
        boost = 0.0
        if "trumpet" in s["label"] or "roar" in s["label"] or "pandemonium" in s["label"]:
            boost = 0.1 if level != "low" else -0.05
        elif "rumble" in s["label"]:
            boost = 0.05 if level in ("low", "medium") else 0.0
        elif "resting" in s["label"]:
            boost = 0.1 if level == "low" else -0.1
        s["confidence"] = float(max(0.0, min(0.99, s["confidence"] + boost)))
    return segments

def summarize(segments: List[Dict[str, Any]], motion_events: List[Dict[str, Any]]) -> Tuple[str, float]:
    if not segments:
        return ("Insufficient evidence to interpret.", 0.4)
    # Choose top label by duration-weighted confidence
    score: Dict[str, float] = {}
    for s in segments:
        score[s["label"]] = score.get(s["label"], 0.0) + (s["end"] - s["start"]) * (0.5 + 0.5 * s["confidence"])
    top = max(score.items(), key=lambda kv: kv[1])[0]
    conf = float(np.median([s["confidence"] for s in segments])) if np is not None else 0.7

    # Compose sentence
    if "greeting chorus" in top:
        msg = "Family reunion chorus—overlapping rumbles with trumpets/roars; strong social excitement."
    elif "let’s‑go" in top:
        msg = "Likely travel initiation—rising rumble from a leader; group may begin moving."
    elif "musth" in top:
        msg = "Musth‑like buzz—deep, buzzy rumble likely from a male; signals status/dominance."
    elif "trumpet" in top:
        msg = "Alarm/excitement—bright trumpet; give space and observe from a distance."
    elif "roar" in top:
        msg = "Defensive/agitated—harsh roar; avoid approaching."
    elif "estrous" in top:
        msg = "Estrous rumble—female receptivity cue; males may show interest."
    elif "contact rumble" in top:
        msg = "Contact/coordination—low rumble to maintain social contact."
    elif "resting" in top:
        msg = "Calm/resting—low acoustic energy and limited motion."
    else:
        msg = "General calling—tonal activity without clear specialized cues."

    # Add notable motion events if present
    notable = [ev["type"] for ev in motion_events if ev["t"] is not None][:2]
    if notable:
        msg += f" Motion notes: {', '.join(notable)}."
    return (msg, conf)

def analyze_media_file(path: Path) -> Dict[str, Any]:
    diagnostics = {
        "ffmpeg": has_ffmpeg(),
        "numpy": bool(np is not None),
        "pillow": bool(Image is not None),
    }

    wav_tmp = UPLOAD_DIR / f"conv_{uuid.uuid4().hex}.wav"
    audio_ok = False
    motion = {"series": [], "events": []}

    try:
        if diagnostics["ffmpeg"]:
            audio_ok = convert_to_wav(path, wav_tmp)
        else:
            if path.suffix.lower() == ".wav":
                wav_tmp = path
                audio_ok = True

        segments: List[Dict[str, Any]] = []
        duration = 0.0

        if audio_ok and np is not None:
            x, sr = load_wav_pcm16(wav_tmp)
            if x is not None:
                segments, duration = classify_audio(x, sr)
        elif audio_ok:
            with contextlib.closing(wave.open(str(wav_tmp), "rb")) as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                duration = n_frames / max(1, sr)
            segments = [{
                "start": 0.0, "end": float(duration), "label": "calling",
                "explanation": "Decoded audio without numpy; fine-grained features unavailable.",
                "confidence": 0.6, "features": {}
            }]
        else:
            segments = [{
                "start": 0.0, "end": 5.0, "label": "uncertain",
                "explanation": "Could not decode audio (install ffmpeg or upload WAV).",
                "confidence": 0.45, "features": {}
            }]
            duration = 5.0

        # Optional motion from video frames
        if diagnostics["ffmpeg"]:
            frames_ok = extract_frames(path, FRAME_DIR, fps=2.0, width=224)
            if frames_ok:
                series = motion_series(FRAME_DIR)
                motion = detect_motion_cues(series, fps=2.0)

        segments = fuse_audio_motion(segments, motion)
        summary_text, overall_conf = summarize(segments, motion.get("events", []))

        return {
            "file": path.name,
            "species": {"label": "African elephant (heuristic)", "confidence": 0.82},
            "segments": segments,
            "summary": summary_text,
            "overall_confidence": float(min(0.99, overall_conf)),
            "diagnostics": diagnostics,
            "duration": duration
        }
    finally:
        if wav_tmp != path:
            try: wav_tmp.unlink(missing_ok=True)
            except Exception: pass
        for f in glob.glob(str(FRAME_DIR / "frame_*.png")):
            try: os.remove(f)
            except Exception: pass

@app.get("/health")
def health():
    return {"status": "ok", "api": "mini++"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_VIDEO.union(SUPPORTED_AUDIO):
        return JSONResponse(status_code=400, content={"error": f"Unsupported type {ext}"})

    tmp_path = UPLOAD_DIR / f"up_{uuid.uuid4().hex}{ext}"
    size = 0
    with tmp_path.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk: break
            size += len(chunk)
            if size > MAX_MB * 1024 * 1024:
                out.close(); tmp_path.unlink(missing_ok=True)
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
