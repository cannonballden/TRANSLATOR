# mini_api.py -- reset-stable v2.2 (ascii-only, richer audio detector)
# FastAPI backend for Elephant Translator:
# - Small files: POST /analyze (direct)
# - Large files: /upload/init + /upload/part + /upload/finish (chunked)
# - Audio heuristics with adaptive thresholds + richer features
# - Video motion (frame-diff), optional optical flow (OpenCV), optional interactions (YOLO)
# - CORS open for GH Pages <-> Codespaces demo. ASCII-only strings.

import os, json, shutil, subprocess, uuid, logging, wave, contextlib, glob, math
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Optional deps (degrade gracefully)
try:
    import numpy as np
except Exception:
    np = None

try:
    from scipy.signal import butter, sosfiltfilt
except Exception:
    butter = None
    sosfiltfilt = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

YOLO_MODEL = None
YOLO_ELE_IDS: Optional[List[int]] = None

log = logging.getLogger("mini_api")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ------------------- Config -------------------
ALLOW_ORIGINS = ["*"]  # open for demo

SMALL_UPLOAD_MAX_MB = int(os.getenv("SMALL_UPLOAD_MAX_MB", "50"))
MAX_MB = int(os.getenv("MAX_MB", "1024"))
CHUNK_MB = int(os.getenv("CHUNK_MB", "5"))
CHUNK_BYTES = CHUNK_MB * 1024 * 1024

SUPPORTED_VIDEO = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".ogv"}
SUPPORTED_AUDIO = {".mp3", ".wav", ".ogg", ".oga", ".flac"}

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads_tmp"
SESS_DIR = UPLOAD_DIR / "sessions"
FRAME_DIR = BASE_DIR / "frames_tmp"
for d in (UPLOAD_DIR, SESS_DIR, FRAME_DIR):
    d.mkdir(exist_ok=True, parents=True)

# ------------------- App -------------------
app = FastAPI(title="Elephant Translator -- reset-stable v2.2 (ascii)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# ------------------- Helpers -------------------
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
    ok, out = _run(["ffmpeg","-y","-i",str(src),"-ac","1","-ar",str(sr),"-vn",str(dst)])
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

def zcr(frame: "np.ndarray") -> float:
    if frame.size < 2:
        return 0.0
    s = np.sign(frame)
    return float(np.mean((s[1:] != s[:-1]).astype(np.float32)))

def butter_band_sos(low, high, sr, order=4):
    if butter is None:
        return None
    nyq = 0.5 * sr
    low = max(1.0, low) / nyq
    high = min(high/nyq, 0.999)
    return butter(order, [low, high], btype='band', output='sos')

def band_rms_fft(frame: "np.ndarray", sr: int, band: Tuple[float,float]) -> float:
    win = np.hamming(len(frame))
    f = np.fft.rfft(frame * win)
    mag = np.abs(f)
    freqs = np.fft.rfftfreq(len(frame), 1.0/sr)
    lo, hi = band
    sel = (freqs >= lo) & (freqs < hi)
    if not np.any(sel):
        return 0.0
    power = float(np.mean((mag[sel])**2))
    return float(np.sqrt(power + 1e-12))

def band_rms(frame: "np.ndarray", sr: int, band: Tuple[float,float]) -> float:
    if butter is not None:
        sos = butter_band_sos(band[0], band[1], sr)
        if sos is not None:
            y = sosfiltfilt(sos, frame)
            return float(np.sqrt(np.mean(y*y) + 1e-12))
    return band_rms_fft(frame, sr, band)

def spectral_stats(frame: "np.ndarray", sr: int, prev_mag: Optional["np.ndarray"]):
    win = np.hamming(len(frame))
    f = np.fft.rfft(frame * win)
    mag = np.abs(f) + 1e-12
    freqs = np.fft.rfftfreq(len(frame), 1.0/sr)

    total = float(np.sum(mag))
    centroid = float(np.sum(freqs * mag) / total)
    flat = float(np.exp(np.mean(np.log(mag))) / (np.mean(mag) + 1e-12))
    roll85_thresh = 0.85 * np.sum(mag)
    cumsum = np.cumsum(mag); roll85_idx = int(np.searchsorted(cumsum, roll85_thresh))
    roll85 = float(freqs[min(roll85_idx, len(freqs)-1)])

    top = np.sort(mag)[-5:]
    tone = float(top.sum() / (total + 1e-12))
    peak_idx = int(np.argmax(mag)); peak_freq = float(freqs[peak_idx])

    low = float(np.sum(mag[(freqs >= 15) & (freqs < 80)]))
    band30 = float(np.sum(mag[(freqs >= 30) & (freqs < 80)]))
    mid = float(np.sum(mag[(freqs >= 100) & (freqs < 300)]))
    upper_mid = float(np.sum(mag[(freqs >= 300) & (freqs < 1000)]))
    high = float(np.sum(mag[(freqs >= 1000) & (freqs < 4000)]))

    low_r = low/total; band30_r = band30/total; mid_r = mid/total
    upper_mid_r = upper_mid/total; high_r = high/total

    if prev_mag is not None:
        d = mag - prev_mag
        flux = float(np.sqrt(np.sum(d*d)) / (len(d) + 1e-9))
    else:
        flux = 0.0

    return {
        "centroid": centroid, "flatness": flat, "roll85": roll85,
        "peak_freq": peak_freq, "tonality": tone,
        "low_r": low_r, "band30_r": band30_r, "mid_r": mid_r,
        "upper_mid_r": upper_mid_r, "high_r": high_r,
    }, mag

def frame_audio(x: "np.ndarray", sr: int, win_s: float = 0.5, hop_s: float = 0.25):
    win = max(1, int(sr * win_s)); hop = max(1, int(sr * hop_s))
    idx = []; start = 0
    while start + win <= len(x):
        idx.append((start, start + win)); start += hop
    if not idx and len(x) > 0:
        idx.append((0, len(x)))
    return idx

def classify_audio_adaptive(x: "np.ndarray", sr: int):
    idx = frame_audio(x, sr, win_s=0.5, hop_s=0.25)
    feats, prev_mag, rms_list = [], None, []
    centroids = []

    for s0, s1 in idx:
        fr = x[s0:s1]
        rms = float(np.sqrt(np.mean(fr*fr) + 1e-12))
        f, prev_mag = spectral_stats(fr, sr, prev_mag)
        f["rms"] = rms
        f["zcr"] = zcr(fr)
        f["rms_low"] = band_rms(fr, sr, (15, 80))
        f["rms_mid"] = band_rms(fr, sr, (100, 300))
        f["rms_high"] = band_rms(fr, sr, (1000, 4000))
        feats.append(f)
        rms_list.append(rms)
        centroids.append(f["centroid"])

    if not feats:
        return [], 0.0, []

    rms_med = float(np.median(rms_list))
    noise_floor = float(np.median(sorted(rms_list)[:max(1, len(rms_list)//5)]))
    def snr_db(r): return 20.0 * math.log10((r + 1e-9) / (noise_floor + 1e-9))

    labels = []
    tonal_marks = []
    duration = len(x) / sr

    for i, ((s0, s1), f) in enumerate(zip(idx, feats)):
        start_t = s0 / sr; end_t = s1 / sr
        snr = snr_db(f["rms"])
        label, conf = "ambient / uncertain", 0.45

        if (snr > 6 and f["high_r"] > 0.35 and f["centroid"] > 1100 and f["zcr"] > 0.05):
            label, conf = "trumpet", min(0.98, 0.60 + 0.4*f["high_r"])
        elif (snr > 4 and f["upper_mid_r"] > 0.33 and f["flatness"] > 0.55 and f["centroid"] > 400):
            label, conf = "roar / rumble-roar", min(0.93, 0.55 + 0.4*f["upper_mid_r"] + 0.2*(f["flatness"]-0.55))
        elif (snr > 3 and f["low_r"] > 0.35 and f["centroid"] < 160 and f["flatness"] < 0.5 and f["peak_freq"] < 120):
            label, conf = "contact rumble", min(0.94, 0.58 + 0.5*f["low_r"])
        elif (snr > 3 and f["low_r"] > 0.28 and f["centroid"] < 260 and
              i >= 2 and (centroids[i-2] < centroids[i-1] < centroids[i])):
            label, conf = "let's-go rumble (matriarch-like)", min(0.90, 0.55 + 0.4*(centroids[i]-centroids[max(0,i-3)]) / 200.0)
        elif (snr > 3 and f["low_r"] > 0.25 and f["flatness"] > 0.62 and f["centroid"] < 220):
            label, conf = "musth-like buzz (male)", min(0.88, 0.52 + 0.5*(f["flatness"]-0.62 + f["low_r"]))
        elif (snr > 3 and 250 <= f["centroid"] <= 600 and (s1-s0)/sr <= 0.8 and f["tonality"] > 0.15):
            label, conf = "estrous rumble (female-like)", 0.74
        elif (f["rms"] < max(noise_floor*1.2, rms_med*0.6)):
            label, conf = "resting / low arousal", min(0.85, 0.6 + (max(noise_floor*1.2, rms_med*0.6) - f["rms"]) * 4.0)
        elif (snr > 2 and f["tonality"] > 0.22 and f["low_r"] > 0.2 and 60 < f["peak_freq"] < 400):
            label, conf = "possible individual-address rumble (uncertain)", 0.64

        if f["tonality"] > 0.22 and 60 < f["peak_freq"] < 500:
            tonal_marks.append((start_t, f["peak_freq"]))

        labels.append({
            "start": float(start_t), "end": float(end_t),
            "label": label,
            "confidence": float(max(0.0, min(conf, 0.99))),
            "explanation": (
                "snr={:.1f}dB, centroid={}Hz, flat={:.2f}, bands(low={:.2f}, 30-80={:.2f}, mid={:.2f}, uMid={:.2f}, high={:.2f}), peak={}Hz, zcr={:.3f}"
                .format(snr, int(f['centroid']), f['flatness'], f['low_r'], f['band30_r'], f['mid_r'], f['upper_mid_r'], f['high_r'], int(f['peak_freq']), f['zcr'])
            ),
            "features": f
        })

    # Merge contiguous same-label
    merged: List[Dict[str, Any]] = []
    for seg in labels:
        if merged and merged[-1]["label"] == seg["label"] and abs(merged[-1]["end"] - seg["start"]) <= 1e-6:
            merged[-1]["end"] = seg["end"]
            merged[-1]["confidence"] = float((merged[-1]["confidence"] + seg["confidence"]) / 2.0)
        else:
            merged.append(seg)

    # Smooth single short islands if flanked by same label
    smoothed: List[Dict[str, Any]] = []
    i = 0
    while i < len(merged):
        if 0 < i < len(merged)-1:
            prevL = merged[i-1]["label"]; nextL = merged[i+1]["label"]
            cur = merged[i]; dur = cur["end"] - cur["start"]
            if prevL == nextL and dur <= 0.35:
                smoothed[-1]["end"] = merged[i+1]["end"]
                smoothed[-1]["confidence"] = float(max(smoothed[-1]["confidence"], merged[i+1]["confidence"]))
                i += 2; continue
        smoothed.append(merged[i]); i += 1
    merged = smoothed

    # Greeting chorus: rumble plus trumpet/roar overlapping or near-overlap
    for i in range(len(merged)):
        for j in range(i+1, len(merged)):
            a, b = merged[i], merged[j]
            if b["start"] > a["end"] + 2.0:
                break
            pair = (
                ("trumpet" in a["label"] or "roar" in a["label"]) and ("rumble" in b["label"])
            ) or (
                ("trumpet" in b["label"] or "roar" in b["label"]) and ("rumble" in a["label"])
            )
            if pair and (min(a["end"], b["end"]) - max(a["start"], b["start"])) > -0.8:
                a["label"] = b["label"] = "greeting chorus (family reunion)"
                a["confidence"] = b["confidence"] = min(0.96, max(a["confidence"], b["confidence"]) + 0.08)

    # File-level findings
    findings = []
    last_end = None; last_rumble = False
    for s in merged:
        is_rumbleish = "rumble" in s["label"]
        if last_rumble and is_rumbleish and last_end is not None:
            gap = s["start"] - last_end
            if 1.0 <= gap <= 5.0:
                findings.append("possible antiphonal exchange (call-and-response timing)")
                break
        last_rumble = is_rumbleish; last_end = s["end"]

    if len(tonal_marks) >= 2:
        tonal_marks.sort()
        for k in range(len(tonal_marks)-1):
            t1, f1 = tonal_marks[k]; t2, f2 = tonal_marks[k+1]
            if 8.0 <= (t2 - t1) <= 20.0 and abs(f2 - f1) <= 25.0:
                findings.append("possible individual-address rumble (repeated narrow peak)")
                break

    return merged, float(duration), findings

# -------- Video extraction & motion --------
def extract_frames(path: Path, out_dir: Path, fps: float = 2.0, width: int = 256) -> bool:
    if not has_ffmpeg():
        return False
    for f in glob.glob(str(out_dir / "frame_*.png")):
        try: os.remove(f)
        except Exception: pass
    ok, out = _run(["ffmpeg","-y","-i",str(path),"-vf",f"fps={fps},scale={width}:-1",str(out_dir/"frame_%05d.png")])
    if not ok:
        log.warning("ffmpeg frames failed: %s", out)
    return ok

def motion_series(out_dir: Path) -> List[float]:
    if Image is None or np is None:
        return []
    files = sorted(Path(out_dir).glob("frame_*.png"))
    if len(files) < 2:
        return []
    vals = []
    prev = np.array(Image.open(files[0]).convert("L"), dtype=np.float32)
    for f in files[1:]:
        cur = np.array(Image.open(f).convert("L"), dtype=np.float32)
        vals.append(float(np.mean(np.abs(cur - prev)) / 255.0))
        prev = cur
    return vals

def optical_flow_series(out_dir: Path, fps: float = 2.0) -> Dict[str, Any]:
    if cv2 is None or np is None:
        return {}
    files = sorted(Path(out_dir).glob("frame_*.png"))
    if len(files) < 2:
        return {}
    mags = []; coh = []
    prev = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
    for f in files[1:]:
        cur = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 1, 15, 3, 5, 1.1, 0)
        vx = flow[...,0]; vy = flow[...,1]
        mag = np.sqrt(vx*vx + vy*vy)
        mags.append(float(np.mean(mag)))
        mvx = float(np.mean(vx)); mvy = float(np.mean(vy))
        denom = float(np.mean(mag) + 1e-6)
        coh.append(float((math.hypot(mvx, mvy)) / denom))
        prev = cur
    events = []
    if len(mags) > 4:
        med = float(np.median(mags)); std = float(np.std(mags))
        run = 0
        for i, (m, c) in enumerate(zip(mags, coh)):
            if m > med + 1.0*std and c > 0.5:
                run += 1
            else:
                if run >= int(1*fps):
                    t = int((i - run/2)/fps)
                    events.append({"t": t, "type": "coherent movement (group moving)", "detail": "approx {}s".format(run/fps)})
                run = 0
        if run >= int(1*fps):
            t = int((len(mags) - run/2)/fps)
            events.append({"t": t, "type": "coherent movement (group moving)", "detail": "approx {}s".format(run/fps)})
    return {"mag": mags, "coh": coh, "events": events}

def detect_motion(vals: List[float], fps: float = 2.0) -> Dict[str, Any]:
    out = {"series": vals, "events": []}
    if not vals or np is None:
        return out
    arr = np.array(vals, dtype=np.float32)
    med = float(np.median(arr)); mean = float(np.mean(arr)); std = float(np.std(arr))

    # Freeze >= 2s
    low_thr = med * 0.4; run = 0
    for i, v in enumerate(arr):
        if v <= low_thr:
            run += 1
        else:
            if run >= int(2*fps):
                t = int((i - run/2)/fps)
                out["events"].append({"t": t, "type": "freeze and listen", "detail": "approx {}s".format(run/fps)})
            run = 0
    if run >= int(2*fps):
        t = int((len(arr) - run/2)/fps)
        out["events"].append({"t": t, "type": "freeze and listen", "detail": "approx {}s".format(run/fps)})

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
                out["events"].append({"t": None, "type": "ear flapping (calm)", "detail": "approx {:.2f} Hz".format(dom_f)})
            elif 0.8 <= dom_f <= 1.5:
                out["events"].append({"t": None, "type": "ear flapping (agitated)", "detail": "approx {:.2f} Hz".format(dom_f)})

    if mean > med * 1.3:
        out["events"].append({"t": None, "type": "high excitement (motion)", "detail": "sustained"})
    return out

# -------- YOLO (optional) --------
def load_yolo() -> bool:
    global YOLO_MODEL, YOLO_ELE_IDS
    if YOLO is None:
        return False
    if YOLO_MODEL is None:
        try:
            YOLO_MODEL = YOLO("yolov8n.pt")
            names = getattr(YOLO_MODEL.model, "names", None) or getattr(YOLO_MODEL, "names", None)
            YOLO_ELE_IDS = None
            if isinstance(names, dict):
                YOLO_ELE_IDS = [i for i,n in names.items() if isinstance(n,str) and n.lower()=="elephant"]
            elif isinstance(names, (list,tuple)):
                YOLO_ELE_IDS = [i for i,n in enumerate(names) if isinstance(n,str) and n.lower()=="elephant"]
        except Exception as e:
            log.warning("YOLO load failed: %s", e)
            YOLO_MODEL = None
            return False
    return YOLO_MODEL is not None

def yolo_detect_frames(out_dir: Path, img_size: int = 640, conf: float = 0.25):
    if not load_yolo():
        return None
    files = sorted(Path(out_dir).glob("frame_*.png"))
    dets_per_frame = []
    for f in files:
        try:
            res = YOLO_MODEL.predict(source=str(f), imgsz=img_size, conf=conf, verbose=False)
            r = res[0]
            boxes = []
            if hasattr(r,"boxes") and r.boxes is not None:
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                for bb, cc, cf in zip(xyxy, cls, confs):
                    if YOLO_ELE_IDS is None or int(cc) in YOLO_ELE_IDS:
                        x1,y1,x2,y2 = [float(v) for v in bb]
                        boxes.append({"xyxy":[x1,y1,x2,y2],"conf":float(cf),"cls":int(cc)})
            dets_per_frame.append(boxes)
        except Exception as e:
            log.warning("YOLO predict failed: %s", e)
            dets_per_frame.append([])
    return dets_per_frame

def iou(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    w,h = max(0.0,ix2-ix1), max(0.0,iy2-iy1)
    inter = w*h
    if inter<=0:
        return 0.0
    areaA=(ax2-ax1)*(ay2-ay1); areaB=(bx2-bx1)*(by2-by1)
    return inter/max(1e-6,(areaA+areaB-inter))

def track_iou(dets_per_frame: List[List[Dict[str,Any]]], iou_thr: float = 0.3):
    tracks_per_frame = []
    next_id = 0
    prev = []
    for dets in dets_per_frame:
        assigned = [-1]*len(dets); used=set()
        for pt in prev:
            best_i, best = -1, 0.0
            for i, d in enumerate(dets):
                if i in used:
                    continue
                sc = iou(pt["xyxy"], d["xyxy"])
                if sc > best:
                    best, best_i = sc, i
            if best >= iou_thr and best_i>=0:
                assigned[best_i]=pt["id"]; used.add(best_i)
        for i in range(len(dets)):
            if assigned[i]==-1:
                assigned[i]=next_id; next_id+=1
        cur=[{"id":assigned[i],"xyxy":dets[i]["xyxy"],"conf":dets[i].get("conf",0.0)} for i in range(len(dets))]
        tracks_per_frame.append(cur); prev=cur
    return tracks_per_frame

def interactions_from_tracks(tracks_per_frame: List[List[Dict[str,Any]]], fps: float = 2.0) -> List[Dict[str,Any]]:
    events: List[Dict[str,Any]] = []
    if not tracks_per_frame:
        return events
    ear_state: Dict[int, Tuple[float,int]] = {}
    ear_run: Dict[int,int] = {}
    close_run: Dict[Tuple[int,int], int] = {}

    for t, cur in enumerate(tracks_per_frame):
        # Ear-spread: width/height ratio spike for >=1s
        for tr in cur:
            x1,y1,x2,y2 = tr["xyxy"]; w=max(1.0,x2-x1); h=max(1.0,y2-y1)
            ratio = w/h
            base,cnt = ear_state.get(tr["id"], (ratio,1))
            base = 0.9*base + 0.1*ratio
            ear_state[tr["id"]] = (base,cnt+1)
            if ratio > base*1.25 and ratio > 1.2:
                ear_run[tr["id"]] = ear_run.get(tr["id"],0)+1
            else:
                if ear_run.get(tr["id"],0) >= int(1.0*fps):
                    events.append({
                        "t": int((t - ear_run[tr["id"]]/2)/fps),
                        "type":"ear spread (threat display)",
                        "detail":"ratio~{:.2f}".format(ratio)
                    })
                ear_run[tr["id"]] = 0

        # Close-contact greeting: head-to-head proximity >= 1.5s
        for i in range(len(cur)):
            for j in range(i+1,len(cur)):
                a,b = cur[i],cur[j]
                ax=0.5*(a["xyxy"][0]+a["xyxy"][2]); ay=0.5*(a["xyxy"][1]+a["xyxy"][3])
                bx=0.5*(b["xyxy"][0]+b["xyxy"][2]); by=0.5*(b["xyxy"][1]+b["xyxy"][3])
                avg_w = 0.5*((a["xyxy"][2]-a["xyxy"][0])+(b["xyxy"][2]-b["xyxy"][0]))
                dist = math.hypot(ax-bx, ay-by)
                key = tuple(sorted((a["id"], b["id"])))
                if dist < 0.6*avg_w:
                    close_run[key] = close_run.get(key,0)+1
                    if close_run[key] == int(1.5*fps):
                        events.append({
                            "t": int((t - close_run[key]/2)/fps),
                            "type":"close-contact trunk greeting (possible twining / trunk-to-mouth)",
                            "detail":"sustained head proximity"
                        })
                else:
                    close_run[key]=0

    # De-dup near-same events
    events.sort(key=lambda e:(e["type"], e.get("t",0)))
    merged=[]
    for e in events:
        if merged and merged[-1]["type"]==e["type"] and abs((merged[-1].get("t",0))-(e.get("t",0)))<=1:
            continue
        merged.append(e)
    return merged

# -------- Fusion & summary --------
def fuse_audio_motion(segments: List[Dict[str, Any]], motion: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not segments:
        return segments
    series = motion.get("series") or []; events = motion.get("events") or []
    if not series or np is None:
        for s in segments:
            s["movement"]={"level":"unknown","notes":[]}
        return segments
    arr = np.array(series, dtype=np.float32)
    med = float(np.median(arr)); q75 = float(np.quantile(arr, 0.75))
    for s in segments:
        mid = int((s["start"] + s["end"]) / 2.0)
        mid = max(0, min(mid, len(arr)-1))
        mv_val = float(arr[mid])
        level = "low" if mv_val < med*0.8 else ("medium" if mv_val < q75*1.1 else "high")
        notes = [ev["type"] for ev in events if ev.get("t") is not None and abs(ev["t"] - mid) <= 1]
        s["movement"]={"level":level,"value":mv_val,"notes":notes}
        boost=0.0; L=s["label"]
        if "trumpet" in L or "roar" in L:
            boost = 0.1 if level!="low" else -0.05
        elif "rumble" in L:
            boost = 0.07 if level in ("low","medium") else 0.02
        elif "resting" in L:
            boost = 0.10 if level=="low" else -0.10
        s["confidence"]=float(max(0.0,min(0.99,s["confidence"]+boost)))
        if "trumpet" in L and level=="high":
            s["label"]="alarm/excitement (trumpet)"
        if "roar" in L and level!="low":
            s["label"]="threat/defensive (roar)"
    return segments

def summarize(segments: List[Dict[str, Any]], motion_events: List[Dict[str, Any]], findings: List[str]):
    if not segments:
        return ("No clear elephant vocalizations detected (audio too quiet or noisy).", 0.4)
    meaningful = [s for s in segments if not s["label"].startswith("ambient")]
    if not meaningful:
        return ("Ambient/uncertain audio; no distinctive elephant calls.", 0.45)

    score={}
    for s in meaningful:
        score[s["label"]] = score.get(s["label"],0.0)+(s["end"]-s["start"])*(0.5+0.5*s["confidence"])
    top = max(score.items(), key=lambda kv: kv[1])[0]
    conf = float(np.median([s["confidence"] for s in meaningful])) if np is not None else 0.7

    if "greeting chorus" in top: msg = "Family reunion chorus: overlapping rumbles with trumpets/roars; strong social excitement."
    elif "let's-go" in top:      msg = "Likely travel initiation: rising rumble; group preparing to move."
    elif "musth" in top:        msg = "Musth-like buzz (male): status/dominance broadcast."
    elif "trumpet" in top:      msg = "Alarm/excitement: bright trumpet; keep distance."
    elif "roar" in top:         msg = "Defensive/agitated: harsh roar; back away."
    elif "estrous" in top:      msg = "Estrous rumble: female receptivity cue."
    elif "contact rumble" in top: msg = "Contact rumble: maintaining social contact/coordination."
    elif "resting" in top:      msg = "Calm/resting: low arousal and limited motion."
    else:                       msg = "General calling: tonal activity without specialized cues."

    notes = [ev["type"] for ev in motion_events if ev.get("t") is not None][:2]
    if notes:
        msg += " Motion: " + ", ".join(notes) + "."
    if findings:
        msg += " Extra: " + ", ".join(findings[:2]) + "."
    return msg, conf

# -------- Core analysis --------
def analyze_media_file(path: Path, include_motion: bool) -> Dict[str, Any]:
    diagnostics = {"ffmpeg": has_ffmpeg(), "numpy": bool(np is not None), "pillow": bool(Image is not None),
                   "opencv": bool(cv2 is not None), "yolo": False, "scipy": bool(butter is not None)}
    wav_tmp = UPLOAD_DIR / ("conv_" + uuid.uuid4().hex + ".wav")
    audio_ok=False; motion={"series": [], "events": []}; duration=0.0
    try:
        if diagnostics["ffmpeg"]:
            audio_ok = convert_to_wav(path, wav_tmp)
        elif path.suffix.lower()==".wav":
            wav_tmp = path; audio_ok=True

        segments=[]; findings=[]
        if audio_ok and np is not None:
            x, sr = load_wav_pcm16(wav_tmp)
            if x is not None and len(x) > 0:
                segments, duration, findings = classify_audio_adaptive(x, sr)
        elif audio_ok:
            with contextlib.closing(wave.open(str(wav_tmp), "rb")) as wf:
                sr=wf.getframerate(); n=wf.getnframes(); duration=n/max(1,sr)
            segments=[{"start":0.0,"end":float(duration),"label":"ambient / uncertain",
                       "explanation":"Decoded audio without numpy; detailed features unavailable.",
                       "confidence":0.5,"features":{}}]
        else:
            segments=[{"start":0.0,"end":5.0,"label":"ambient / uncertain",
                       "explanation":"Could not decode audio (install ffmpeg or upload WAV).",
                       "confidence":0.45,"features":{}}]
            duration=5.0

        if include_motion and diagnostics["ffmpeg"]:
            if extract_frames(path, FRAME_DIR, fps=2.0, width=256):
                series = motion_series(FRAME_DIR)
                motion = detect_motion(series, fps=2.0)
                oflow = optical_flow_series(FRAME_DIR, fps=2.0)
                if oflow:
                    motion["oflow"]={"mean_mag": oflow.get("mag", []), "coherence": oflow.get("coh", [])}
                    motion["events"].extend(oflow.get("events", []))
                dets = yolo_detect_frames(FRAME_DIR)
                if dets is not None:
                    diagnostics["yolo"]=True
                    tracks = track_iou(dets)
                    inter = interactions_from_tracks(tracks, fps=2.0)
                    motion["events"].extend(inter)

        segments = fuse_audio_motion(segments, motion)
        summary_text, overall_conf = summarize(segments, motion.get("events", []), findings)

        elephant_likely = float(np.mean([1.0 if (("rumble" in s["label"]) or ("chorus" in s["label"])) else 0.0 for s in segments])) if np is not None else 0.6
        species_conf = min(0.95, 0.55 + 0.45*elephant_likely)

        return {
            "file": path.name,
            "species": {"label":"African elephant (heuristic)", "confidence": float(species_conf)},
            "segments": segments,
            "summary": summary_text,
            "overall_confidence": float(min(0.99, overall_conf)),
            "diagnostics": diagnostics,
            "duration": duration,
            "findings": findings,
            "motion_events": motion.get("events", []),
            "motion_series_len": len(motion.get("series", [])),
        }
    finally:
        if wav_tmp != path:
            try: wav_tmp.unlink(missing_ok=True)
            except Exception: pass
        for f in glob.glob(str(FRAME_DIR/"frame_*.png")):
            try: os.remove(f)
            except Exception: pass

# ------------------- API -------------------
@app.get("/health")
def health():
    return {"status":"ok","api":"reset-stable v2.2 + richer audio (ascii)"}

@app.get("/config")
def config():
    return {"small_upload_max_mb": SMALL_UPLOAD_MAX_MB, "max_mb": MAX_MB, "chunk_mb": CHUNK_MB}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), include_motion: bool = True):
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_VIDEO.union(SUPPORTED_AUDIO):
        return JSONResponse(status_code=400, content={"error": f"Unsupported type {ext}"})
    tmp = UPLOAD_DIR / ("direct_" + uuid.uuid4().hex + ext)
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
    data = await request.body()
    if not data:
        return JSONResponse(status_code=400, content={"error":"empty chunk"})
    part_path = parts / ("part_{:05d}.bin".format(index))
    with part_path.open("wb") as f:
        f.write(data)
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
    if total_chunks <= 0:
        total_chunks = len(list(parts.glob("part_*.bin")))
    for i in range(total_chunks):
        if not (parts / ("part_{:05d}.bin".format(i))).exists():
            return JSONResponse(status_code=400, content={"error": "missing chunk {}".format(i)})
    final_path = UPLOAD_DIR / ("chunked_" + upload_id + ext)
    with final_path.open("wb") as out:
        for i in range(total_chunks):
            p = parts / ("part_{:05d}.bin".format(i))
            with p.open("rb") as f:
                shutil.copyfileobj(f, out)
    try:
        res = analyze_media_file(final_path, include_motion=include_motion)
        return JSONResponse(content=res)
    except Exception as e:
        log.exception("Analysis failed")
        return JSONResponse(status_code=500, content={"error": f"Analysis failed: {e}"})
    finally:
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
