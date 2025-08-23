#!/usr/bin/env python3
# training/train_classifier.py
# Trains a scikit-learn model on segment WAVs listed in the latest manifest.
# Uses feature recipe that matches mini_api; prefers manual_label when present.

import os, json, math, argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

REPO = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO / "training"
SEG_DIR = TRAIN_DIR / "segments_audio"
MAN_DIR = TRAIN_DIR / "manifests"
MODEL_DIR = REPO / "models"
MODEL_DIR.mkdir(exist_ok=True)

def band_rms(x, sr, lo, hi):
    win = np.hamming(len(x))
    f = np.fft.rfft(x * win)
    mag = np.abs(f) + 1e-12
    freqs = np.fft.rfftfreq(len(x), 1.0/sr)
    sel = (freqs >= lo) & (freqs < hi)
    if not np.any(sel):
        return 0.0
    power = np.mean((mag[sel])**2)
    return float(np.sqrt(power + 1e-12))

def feats_for_clip(wav_path: Path) -> List[float]:
    x, sr = sf.read(str(wav_path), always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    rms = float(np.sqrt(np.mean(x*x) + 1e-12))
    win = max(1, int(sr*0.5)); hop = max(1, int(sr*0.25))
    rows = []
    for i in range(0, len(x)-win+1, hop):
        fr = x[i:i+win]
        winw = np.hamming(len(fr))
        F = np.fft.rfft(fr * winw)
        mag = np.abs(F) + 1e-12
        freqs = np.fft.rfftfreq(len(fr), 1.0/sr)
        total = float(np.sum(mag))
        centroid = float(np.sum(freqs*mag)/total)
        flat = float(np.exp(np.mean(np.log(mag)))/(np.mean(mag)+1e-12))
        roll85 = float(freqs[min(int(np.searchsorted(np.cumsum(mag), 0.85*np.sum(mag))), len(freqs)-1)])
        peak = float(freqs[int(np.argmax(mag))])
        tone = float(np.sort(mag)[-5:].sum()/(total+1e-12))
        low_r = float(np.sum(mag[(freqs>=15)&(freqs<80)])/total)
        b30_r = float(np.sum(mag[(freqs>=30)&(freqs<80)])/total)
        mid_r = float(np.sum(mag[(freqs>=100)&(freqs<300)])/total)
        umid_r = float(np.sum(mag[(freqs>=300)&(freqs<1000)])/total)
        high_r = float(np.sum(mag[(freqs>=1000)&(freqs<4000)])/total)
        zcr = float(np.mean(np.sign(fr)[1:] != np.sign(fr)[:-1]))
        rows.append([
            rms, zcr, centroid, flat, roll85, peak, tone,
            low_r, b30_r, mid_r, umid_r, high_r,
            band_rms(fr, sr, 15, 80),
            band_rms(fr, sr, 100, 300),
            band_rms(fr, sr, 1000, 4000)
        ])
    if not rows:
        return [rms]+[0.0]*14
    return list(np.mean(np.array(rows, dtype=np.float32), axis=0))

def normalize_label(text: str) -> str:
    L = (text or "").lower()
    if "trumpet" in L: return "trumpet"
    if "roar" in L: return "roar"
    if "musth" in L: return "musth-like"
    if "let's-go" in L or "lets-go" in L: return "lets-go"
    if "estrous" in L: return "estrous"
    if "contact-rumble" in L or "contact rumble" in L: return "contact-rumble"
    if "greeting" in L: return "greeting-chorus"
    if "resting" in L: return "resting"
    if "rumble" in L: return "rumble"
    return "other"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default=None, help="path to manifest CSV (default: latest)")
    args = ap.parse_args()

    if args.manifest:
        mf = Path(args.manifest)
    else:
        mfs = sorted(MAN_DIR.glob("manifest_*.csv"))
        if not mfs:
            print("No manifests found. Run auto_ingest.py first.")
            return
        mf = mfs[-1]
    print("Using manifest:", mf)

    df = pd.read_csv(mf)
    X = []; y = []
    for _, r in df.iterrows():
        seg = Path(str(r["segment"]))
        if not seg.exists(): continue
        if "manual_label" in r and isinstance(r["manual_label"], str) and r["manual_label"].strip():
            label = normalize_label(r["manual_label"])
        else:
            label = normalize_label(str(r.get("pred_label","")))
        try:
            feat = feats_for_clip(seg)
        except Exception:
            continue
        X.append(feat); y.append(label)

    if not X:
        print("No training data extracted.")
        return

    X = np.array(X, dtype=np.float32); y = np.array(y)
    classes = sorted(list(set(y)))
    print("Labels:", classes, "N=", len(y))

    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weight = {c: w for c, w in zip(classes, cw)}

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    clf = LogisticRegression(max_iter=300, class_weight=class_weight, multi_class="auto")
    clf.fit(Xtr_s, ytr)

    from sklearn.metrics import classification_report, confusion_matrix
    Xte_s = scaler.transform(Xte)
    yhat = clf.predict(Xte_s)
    print("Report:\n", classification_report(yte, yhat, digits=3))
    print("Confusion matrix:\n", confusion_matrix(yte, yhat, labels=classes))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(clf, MODEL_DIR/"call_model.joblib")
    joblib.dump(scaler, MODEL_DIR/"scaler.joblib")
    (MODEL_DIR/"labels.json").write_text(json.dumps(classes))
    print("Saved model to", MODEL_DIR)

if __name__ == "__main__":
    main()
