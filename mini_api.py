# mini_api.py — enhanced FastAPI backend with chunked uploads + richer heuristics
# - Large files via /upload/* (chunked) + small files via /analyze
# - Audio features: centroid, bands, rolloff, flatness, flux, tonality, peak, centroid slope
# - Motion cues via ffmpeg frame diffs (optional)
# - Conservative interpretations + clear uncertainty
# - Graceful fallbacks if numpy/Pillow/ffmpeg aren't available

import os, json, shutil, subprocess, uuid, logging, wave, contextlib, glob
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

# Optional deps (graceful degradation)
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

# ---------- Config ----------
PAGES_ORIGIN = os.getenv("PAGES_ORIGIN", "https://cannonballden.github.io")
SMALL_UPLOAD_MAX_MB = int(os.getenv("SMALL_UPLOAD_MAX_MB", "25"))  # direct /analyze limit
MAX_MB = int(os.getenv("MAX_MB", "512"))  # absolute upper bound (chunked or not)
CHUNK_MB = int(os.getenv("CHUNK_MB", "5"))  # chunk size announced by /upload/init
CHUNK_BYTES = CHUNK_MB * 1024 * 1024

SUPPORTED_VIDEO = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".ogv"}
SUPPORTED_AUDIO = {".mp3", ".wav", ".ogg", ".oga", ".flac"}

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads_tmp"
SESS_DIR = UPLOAD_DIR / "sessions"         # for chunked uploads
FRAME_DIR = BASE_DIR / "frames_tmp"        # for motion analysis
for d in (UPLOAD_DIR, SESS_DIR, FRAME_DIR):
    d.mkdir(exist_ok=True, parents=True)

# ---------- App ----------
app = FastAPI(title="Elephant Translator — API (large-file + heuristics)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[PAGES_ORIGIN],  # origin = scheme+host (no path)
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# ---------- Utilities ----------
def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def _run(cmd: List[str]) -> Tuple[bool, str]:
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, text=True)
        return (res.returncode == 0, res.stdout)
    except Exception as e:
        return (False, str(e))

def convert_to_wav(src: Path, dst: Path, sr: int = 16000) -> bool:
    if not has_ffmpeg():
        return False
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", str(sr), "-vn", str(dst)]
    ok, out = _run(cmd)
    if not ok:
        log.warning("ffmpeg failed: %s", out)
    return ok

def load_wav_pcm16(path: Path) -> Tuple[Optional["np.ndarray"], int]:
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

def frame_audio(x: "np.ndarray", sr: int, win_s: float = 0.5, hop_s: float = 0.25):
    win = max(1, int(sr * win_s)); hop = max(1, int(sr * hop_s))
    idx = []; start = 0
    while start + win <= len(x):
        idx.append((start, start + win)); start += hop
    if not idx and len(x) > 0: idx.append((0, len(x)))
    return idx

def spectral_flatness(mag: "np.ndarray") -> float:
    eps = 1e-9
    gm = float(np.exp(np.mean(np.log(mag + eps))))
    am = float(np.mean(mag + eps))
    return gm / am

def spectral_rolloff(freqs: "np.ndarray", mag: "np.ndarray", pct: float = 0.85) -> float:
    cumsum = np.cumsum(mag); thresh = pct * cumsum[-1]
    idx = int(np.searchsorted(cumsum, thresh))
    return float(freqs[min(idx, len(freqs)-1)])

def spectral_flux(prev_mag: Optional["np.ndarray"], mag: "np.ndarray") -> float:
    if prev_mag is None: return 0.0
    d = mag - prev_mag
    return float(np.sqrt(np.sum(d * d)) / (len(d) + 1e-9))

def tonal_peakness(mag: "np.ndarray") -> float:
    if len(mag) < 8: return 0.0
    top = float(np.sort(mag)[-5:].sum())
    return float(top / (mag.sum() + 1e-9))

def spectral_features(frame: "np.ndarray", sr: int, prev_mag: Optional["np.ndarray"]):
    w = np.hamming(len(frame))
    f = np.fft.rfft(frame * w); mag = np.abs(f) + 1e-9
    freqs = np.fft.rfftfreq(len(frame), 1.0 / sr)
    total = mag.sum()
    centroid = float((freqs * mag).sum() / total)
    flat = spectral_flatness(mag)
    roll85 = spectral_rolloff(freqs, mag, pct=0.85)
    flux = spectral_flux(prev_mag, mag)
    peak_idx = int(np.argmax(mag)); peak_freq = float(freqs[peak_idx])
    tone = tonal_peakness(mag)
    low = float(mag[(freqs < 80)].sum())
    band30 = float(mag[(freqs >= 30) & (freqs < 80)].sum())
    mid = float(mag[(freqs >= 100) & (freqs < 300)].sum())
    upper_mid = float(mag[(freqs >= 300) & (freqs < 1000)].sum())
    high = float(mag[(freqs >= 1000) & (freqs < 4000)].sum())
    low_r = low / total; band30_r = band30/total; mid_r = mid/total
    upper_mid_r = upper_mid/total; high_r = high/total
    rms = float(np.sqrt(np.mean(frame ** 2)))
    feat = {
        "rms": rms, "centroid": centroid, "flatness": flat, "roll85": roll85, "flux": flux,
        "peak_freq": peak_freq, "tonality": tone,
        "low_r": low_r, "band30_r": band30_r, "mid_r": mid_r,
        "upper_mid_r": upper_mid_r, "high_r": high_r
    }
    return feat, mag

def classify_audio(x: "np.ndarray", sr: int):
    """Return (segments, duration, findings) where findings are global notes (e.g., antiphony)."""
    idx = frame_audio(x, sr, win_s=0.6, hop_s=0.3)
    feats = []; prev_mag = None
    for (s, e) in idx:
        f, prev_mag = spectral_features(x[s:e], sr, prev_mag)
        feats.append(f)
    if not feats:
        return [], 0.0, []

    rms_vals = np.array([f["rms"] for f in feats], dtype=np.float32)
    energy_med = float(np.median(rms_vals))
    energy_hi = float(np.quantile(rms_vals, 0.75))
    cent = np.array([f["centroid"] for f in feats], dtype=np.float32)
    slope = np.zeros_like(cent)
    if len(cent) >= 3:
        slope[1:-1] = (cent[2:] - cent[:-2]) / 2.0
    dur = len(x) / sr

    # Label each frame
    labels = []
    tonal_marks = []  # (t, peak_freq) for name-like / antiphonal checks
    for i, (samp, (s0, s1)) in enumerate(zip(feats, idx)):
        start_t = s0 / sr; end_t = s1 / sr
        label, conf, reasons = "calling", 0.55, []

        if samp["rms"] > max(energy_hi, energy_med * 1.6) and samp["high_r"] > 0.35 and samp["centroid"] > 1200:
            label = "trumpet"; conf = min(0.97, 0.65 + 0.4*samp["high_r"] + 0.0001*(samp["centroid"]-1200))
            reasons.append("bright high-band energy & high centroid")
        elif samp["rms"] > energy_med * 0.8 and samp["low_r"] > 0.33 and samp["centroid"] < 180 and samp["flatness"] < 0.5:
            label = "contact rumble"; conf = min(0.92, 0.6 + 0.5*samp["low_r"])
            reasons.append("strong low-frequency energy, smooth spectrum")
        elif samp["rms"] > energy_med * 0.9 and samp["low_r"] > 0.28 and slope[i] > 30 and samp["centroid"] < 260:
            label = "let’s‑go rumble (matriarch)"; conf = min(0.9, 0.55 + 0.5*min(1.0, slope[i]/120.0))
            reasons.append("low band + rising centroid/energy")
        elif samp["rms"] > energy_med * 0.9 and samp["low_r"] > 0.25 and samp["flatness"] > 0.6 and samp["centroid"] < 220:
            label = "musth‑like buzz (male)"; conf = min(0.86, 0.5 + 0.6*(samp["flatness"] - 0.6 + samp["low_r"]))
            reasons.append("low centroid with buzzy flat spectrum")
        elif samp["rms"] > energy_med and samp["upper_mid_r"] > 0.28 and samp["flatness"] > 0.55:
            label = "roar / rumble‑roar"; conf = min(0.9, 0.55 + 0.5*samp["upper_mid_r"])
            reasons.append("harsh mid/upper-mid energy")
        elif 250 <= samp["centroid"] <= 600 and samp["rms"] > energy_med * 0.8 and (s1 - s0) / sr <= 0.75:
            label = "estrous rumble (female)"; conf = 0.72
            reasons.append("short, higher‑pitched rumble")
        elif samp["rms"] < energy_med * 0.6:
            label = "resting / low arousal"; conf = min(0.82, 0.6 + (energy_med*0.6 - samp["rms"])*3.0)
            reasons.append("low overall energy")
        else:
            if samp["tonality"] > 0.22 and samp["low_r"] > 0.2 and 80 < samp["peak_freq"] < 400:
                label = "possible individual‑address rumble (uncertain)"; conf = 0.62
                reasons.append("tonal narrowband peak")
        if samp["tonality"] > 0.22 and 60 < samp["peak_freq"] < 500:
            tonal_marks.append((start_t, samp["peak_freq"]))

        labels.append({
            "start": float(start_t), "end": float(end_t),
            "label": label, "confidence": float(max(0.0, min(conf, 0.99))),
            "explanation": (
                f"rms={samp['rms']:.3f}, centroid={samp['centroid']:.0f}Hz, flat={samp['flatness']:.2f}, "
                f"bands(low={samp['low_r']:.2f}, 30-80={samp['band30_r']:.2f}, mid={samp['mid_r']:.2f}, "
                f"uMid={samp['upper_mid_r']:.2f}, high={samp['high_r']:.2f}), "
                f"peak={samp['peak_freq']:.0f}Hz, slope={slope[i]:.1f}"
            ),
            "features": samp
        })

    # Merge contiguous same-label segments
    merged = []
    for seg in labels:
        if merged and merged[-1]["label"] == seg["label"] and abs(merged[-1]["end"] - seg["start"]) < 1e-6:
            merged[-1]["end"] = seg["end"]
            merged[-1]["confidence"] = float((merged[-1]["confidence"] + seg["confidence"]) / 2.0)
        else:
            merged.append(seg)

    # Greeting chorus: overlaps (rumble + trumpet/roar) within ~2s
    for i in range(len(merged)):
        for j in range(i+1, len(merged)):
            a, b = merged[i], merged[j]
            if b["start"] > a["end"] + 2.0: break
            chorus_pair = (
                ("trumpet" in a["label"] or "roar" in a["label"]) and ("rumble" in b["label"])
            ) or (
                ("trumpet" in b["label"] or "roar" in b["label"]) and ("rumble" in a["label"])
            )
            # rough overlap window
            if chorus_pair and min(a["end"], b["end"]) - max(a["start"], b["start"]) > -1.0:
                a["label"] = "greeting chorus (family reunion)"
                b["label"] = "greeting chorus (family reunion)"
                a["confidence"] = b["confidence"] = min(0.95, max(a["confidence"], b["confidence"]) + 0.05)

    findings = []

    # Possible antiphonal exchange: two contact/let’s-go segments with ~1–5s gap
    last_end = None
    last_is_rumble = False
    for s in merged:
        is_rumbleish = "rumble" in s["label"]
        if last_is_rumble and is_rumbleish and last_end is not None:
            gap = s["start"] - last_end
            if 1.0 <= gap <= 5.0:
                findings.append("possible antiphonal exchange (call-and-response timing)")
                break
        last_is_rumble = is_rumbleish
        last_end = s["end"]

    # Possible individual-address: repeated tonal peak near same freq with 8–20s separation
    if len(tonal_marks) >= 2:
        tonal_marks.sort()
        for i in range(len(tonal_marks)-1):
            t1, f1 = tonal_marks[i]; t2, f2 = tonal_marks[i+1]
            if 8.0 <= (t2 - t1) <= 20.0 and abs(f2 - f1) <= 25.0:
                findings.append("possible individual-address rumble (repeated narrow peak)")
                break

    return merged, float(dur), findings

def extract_frames(path: Path, out_dir: Path, fps: float = 2.0, width: int = 224) -> bool:
    if not has_ffmpeg(): return False
    for f in glob.glob(str(out_dir / "frame_*.png")):
        try: os.remove(f)
        except Exception: pass
    ok, out = _run(["ffmpeg","-y","-i",str(path),"-vf",f"fps={fps},scale={width}:-1,format=gray",str(out_dir/"frame_%05d.png")])
    if not ok: log.warning("ffmpeg frames failed: %s", out)
    return ok

def motion_series(out_dir: Path) -> List[float]:
    if Image is None or np is None: return []
    files = sorted(Path(out_dir).glob("frame_*.png"))
    if len(files) < 2: return []
    vals = []
    prev = np.array(Image.open(files[0]), dtype=np.float32)
    for f in files[1:]:
        cur = np.array(Image.open(f), dtype=np.float32)
        vals.append(float(np.mean(np.abs(cur - prev)) / 255.0))
        prev = cur
    return vals

def detect_motion(vals: List[float], fps: float = 2.0) -> Dict[str, Any]:
    out = {"series": vals, "events": []}
    if not vals or np is None: return out
    arr = np.array(vals, dtype=np.float32)
    med = float(np.median(arr)); mean = float(np.mean(arr)); std = float(np.std(arr))

    # Freeze ≥2s
    low_thr = med * 0.4; run = 0
    for i, v in enumerate(arr):
        if v <= low_thr: run += 1
        else:
            if run >= int(2*fps):
                t = int((i - run/2)/fps)
                out["events"].append({"t": t, "type": "freeze and listen", "detail": f"~{run/fps:.1f}s low motion"})
            run = 0
    if run >= int(2*fps):
        t = int((len(arr) - run/2)/fps)
        out["events"].append({"t": t, "type": "freeze and listen", "detail": f"~{run/fps:.1f}s low motion"})

    # Stomp spikes
    if std > 0:
        spikes = np.where(arr > med + 2.5*std)[0]
        for i in spikes:
            out["events"].append({"t": int(i/fps), "type": "foot stomp / seismic", "detail": "sudden spike"})

    # Ear flapping periodicity
    if len(arr) >= 8:
        a = arr - np.mean(arr)
        spec = np.abs(np.fft.rfft(a))
        freqs = np.fft.rfftfreq(len(a), d=1.0/fps)
        if len(spec) > 1:
            dom_idx = int(np.argmax(spec[1:]) + 1)
            dom_f = float(freqs[dom_idx])
            if 0.2 <= dom_f <= 0.6:
                out["events"].append({"t": None, "type": "ear flapping (calm)", "detail": f"~{dom_f:.2f} Hz"})
            elif 0.8 <= dom_f <= 1.5:
                out["events"].append({"t": None, "type": "ear flapping (agitated)", "detail": f"~{dom_f:.2f} Hz"})

    if mean > med * 1.3:
        out["events"].append({"t": None, "type": "high excitement (motion)", "detail": "sustained elevated motion"})
    return out

def fuse_audio_motion(segments: List[Dict[str, Any]], motion: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not segments: return segments
    series = motion.get("series") or []; events = motion.get("events") or []
    if not series or np is None:
        for s in segments: s["movement"] = {"level": "unknown", "notes": []}
        return segments
    arr = np.array(series, dtype=np.float32)
    med = float(np.median(arr)); q75 = float(np.quantile(arr, 0.75))
    for s in segments:
        mid = int((s["start"] + s["end"]) / 2.0)
        mv_val = float(arr[min(mid, len(arr)-1)])
        level = "low" if mv_val < med*0.8 else ("medium" if mv_val < q75*1.1 else "high")
        notes = [ev["type"] for ev in events if ev["t"] is not None and abs(ev["t"] - mid) <= 1]
        s["movement"] = {"level": level, "value": mv_val, "notes": notes}
        # modest fusion
        boost = 0.0
        L = s["label"]
        if "trumpet" in L or "roar" in L: boost = 0.1 if level != "low" else -0.05
        elif "rumble" in L: boost = 0.05 if level in ("low","medium") else 0.0
        elif "resting" in L: boost = 0.1 if level == "low" else -0.1
        s["confidence"] = float(max(0.0, min(0.99, s["confidence"] + boost)))
    return segments

def summarize(segments: List[Dict[str, Any]], motion_events: List[Dict[str, Any]], findings: List[str]):
    if not segments: return ("Insufficient evidence to interpret.", 0.4)
    score: Dict[str, float] = {}
    for s in segments:
        score[s["label"]] = score.get(s["label"], 0.0) + (s["end"] - s["start"]) * (0.5 + 0.5*s["confidence"])
    top = max(score.items(), key=lambda kv: kv[1])[0]
    conf = float(np.median([s["confidence"] for s in segments])) if np is not None else 0.7

    if "greeting chorus" in top:
        msg = "Family reunion chorus—overlapping rumbles with trumpets/roars; strong social excitement."
    elif "let’s‑go" in top:
        msg = "Likely travel initiation—rising rumble from a leader; group may begin moving."
    elif "musth" in top:
        msg = "Musth‑like buzz—deep buzzy rumble (male); signals status/dominance."
    elif "trumpet" in top:
        msg = "Alarm/excitement—bright trumpet; give space."
    elif "roar" in top:
        msg = "Defensive/agitated—harsh roar; keep distance."
    elif "estrous" in top:
        msg = "Estrous rumble—female receptivity cue."
    elif "contact rumble" in top:
        msg = "Contact/coordination—low rumble to keep in touch."
    elif "resting" in top:
        msg = "Calm/resting—low energy and limited motion."
    else:
        msg = "General calling—tonal activity without specialized cues."

    notes = [ev["type"] for ev in motion_events if ev["t"] is not None][:2]
    if notes: msg += f" Motion notes: {', '.join(notes)}."
    if findings:
        msg += f" Extra cues: {', '.join(findings[:2])}."
    return msg, conf

def analyze_media_file(path: Path, include_motion: bool) -> Dict[str, Any]:
    diagnostics = {"ffmpeg": has_ffmpeg(), "numpy": bool(np is not None), "pillow": bool(Image is not None)}

    wav_tmp = UPLOAD_DIR / f"conv_{uuid.uuid4().hex}.wav"
    audio_ok = False; motion = {"series": [], "events": []}
    duration = 0.0

    try:
        if diagnostics["ffmpeg"]:
            audio_ok = convert_to_wav(path, wav_tmp)
        elif path.suffix.lower() == ".wav":
            wav_tmp = path; audio_ok = True

        segments: List[Dict[str, Any]] = []; findings: List[str] = []
        if audio_ok and np is not None:
            x, sr = load_wav_pcm16(wav_tmp)
            if x is not None:
                segments, duration, findings = classify_audio(x, sr)
        elif audio_ok:
            with contextlib.closing(wave.open(str(wav_tmp), "rb")) as wf:
                sr = wf.getframerate(); n_frames = wf.getnframes()
                duration = n_frames / max(1, sr)
            segments = [{
                "start": 0.0, "end": float(duration), "label": "calling",
                "explanation": "Decoded audio without numpy; detailed features unavailable.",
                "confidence": 0.6, "features": {}
            }]
        else:
            segments = [{
                "start": 0.0, "end": 5.0, "label": "uncertain",
                "explanation": "Could not decode audio (install ffmpeg or upload WAV).",
                "confidence": 0.45, "features": {}
            }]
            duration = 5.0

        if include_motion and diagnostics["ffmpeg"]:
            if extract_frames(path, FRAME_DIR, fps=2.0, width=224):
                series = motion_series(FRAME_DIR)
                motion = detect_motion(series, fps=2.0)

        segments = fuse_audio_motion(segments, motion)
        summary_text, overall_conf = summarize(segments, motion.get("events", []), findings)

        return {
            "file": path.name,
            "species": {"label": "African elephant (heuristic)", "confidence": 0.84},
            "segments": segments,
            "summary": summary_text,
            "overall_confidence": float(min(0.99, overall_conf)),
            "diagnostics": diagnostics,
            "duration": duration,
            "findings": findings
        }
    finally:
        if wav_tmp != path:
            try: wav_tmp.unlink(missing_ok=True)
            except Exception: pass
        for f in glob.glob(str(FRAME_DIR / "frame_*.png")):
            try: os.remove(f)
            except Exception: pass

# ---------- API: health/config ----------
@app.get("/health")
def health():
    return {"status": "ok", "api": "large-file+heuristics"}

@app.get("/config")
def config():
    return {
        "small_upload_max_mb": SMALL_UPLOAD_MAX_MB,
        "max_mb": MAX_MB,
        "chunk_mb": CHUNK_MB
    }

# ---------- API: small file (direct) ----------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...), include_motion: bool = True):
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_VIDEO.union(SUPPORTED_AUDIO):
        return JSONResponse(status_code=400, content={"error": f"Unsupported type {ext}"})
    # size guard for direct route (browser may still send large)
    tmp = UPLOAD_DIR / f"direct_{uuid.uuid4().hex}{ext}"
    size = 0
    with tmp.open("wb") as out:
        while True:
            chunk = await file.read(1024*1024)
            if not chunk: break
            size += len(chunk)
            if size > SMALL_UPLOAD_MAX_MB * 1024 * 1024:
                out.close(); tmp.unlink(missing_ok=True)
                return JSONResponse(status_code=413, content={"error": f"File too large for direct upload (> {SMALL_UPLOAD_MAX_MB} MB). Use chunked mode."})
            out.write(chunk)
    try:
        res = analyze_media_file(tmp, include_motion=include_motion)
        return JSONResponse(content=res)
    except Exception as e:
        log.exception("Analysis failed")
        return JSONResponse(status_code=500, content={"error": f"Analysis failed: {e}"})
    finally:
        tmp.unlink(missing_ok=True)

# ---------- API: chunked upload (large files) ----------
# layout: sessions/{upload_id}/parts/part_00000.bin + meta.json
@app.post("/upload/init")
async def upload_init(req: Request):
    try:
        body = await req.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error":"Invalid JSON"})
    filename = body.get("filename") or "unknown"
    filesize = int(body.get("filesize") or 0)
    if filesize <= 0 or (filesize > MAX_MB*1024*1024):
        return JSONResponse(status_code=400, content={"error": f"filesize must be 1..{MAX_MB} MB"})
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_VIDEO.union(SUPPORTED_AUDIO):
        return JSONResponse(status_code=400, content={"error": f"Unsupported type {ext}"})
    upload_id = uuid.uuid4().hex
    sess = SESS_DIR / upload_id
    parts = sess / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    meta = {"filename": filename, "filesize": filesize, "ext": ext, "received": {}}
    (sess / "meta.json").write_text(json.dumps(meta))
    return {"upload_id": upload_id, "chunk_bytes": CHUNK_BYTES}

@app.post("/upload/part")
async def upload_part(request: Request, upload_id: str, index: int):
    sess = SESS_DIR / upload_id
    parts = sess / "parts"
    meta_p = sess / "meta.json"
    if not meta_p.exists():
        return JSONResponse(status_code=404, content={"error":"upload session not found"})
    # read raw body
    data = await request.body()
    if not data:
        return JSONResponse(status_code=400, content={"error":"empty chunk"})
    part_path = parts / f"part_{index:05d}.bin"
    with part_path.open("wb") as f:
        f.write(data)
    # update meta
    meta = json.loads(meta_p.read_text())
    meta["received"][str(index)] = len(data)
    meta_p.write_text(json.dumps(meta))
    return {"ok": True, "index": index, "size": len(data)}

@app.post("/upload/finish")
async def upload_finish(upload_id: str, include_motion: bool = True, total_chunks: int = 0):
    sess = SESS_DIR / upload_id
    parts = sess / "parts"
    meta_p = sess / "meta.json"
    if not meta_p.exists():
        return JSONResponse(status_code=404, content={"error":"upload session not found"})
    meta = json.loads(meta_p.read_text())
    filename = meta["filename"]; ext = meta["ext"]
    # verify parts
    if total_chunks <= 0:
        # try to infer
        total_chunks = len(list(parts.glob("part_*.bin")))
    for i in range(total_chunks):
        if not (parts / f"part_{i:05d}.bin").exists():
            return JSONResponse(status_code=400, content={"error": f"missing chunk {i}"})
    # assemble
    final_path = UPLOAD_DIR / f"chunked_{upload_id}{ext}"
    with final_path.open("wb") as out:
        for i in range(total_chunks):
            p = parts / f"part_{i:05d}.bin"
            with p.open("rb") as f:
                shutil.copyfileobj(f, out)
    # analyze
    try:
        res = analyze_media_file(final_path, include_motion=include_motion)
        return JSONResponse(content=res)
    except Exception as e:
        log.exception("Analysis failed")
        return JSONResponse(status_code=500, content={"error": f"Analysis failed: {e}"})
    finally:
        # cleanup
        try: final_path.unlink(missing_ok=True)
        except Exception: pass
        for f in parts.glob("part_*.bin"):
            try: f.unlink()
            except Exception: pass
        try: meta_p.unlink()
        except Exception: pass
        try: parts.rmdir()
        except Exception: pass
        try: sess.rmdir()
        except Exception: pass
