#!/usr/bin/env python3
# Trains a scikit-learn model on extracted segments; saves into models/

import json
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

REPO = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO / "training"
SEG_DIR = TRAIN_DIR / "segments_audio"
MAN_DIR = TRAIN_DIR / "manifests"
MODEL_DIR = REPO / "models"; MODEL_DIR.mkdir(exist_ok=True)

def band_rms(x, sr, lo, hi):
    win = np.hamming(len(x)); f = np.fft.rfft(x*win); mag = np.abs(f)+1e-12
    fr = np.fft.rfftfreq(len(x), 1.0/sr); sel = (fr>=lo)&(fr<hi)
    return float(np.sqrt(np.mean((mag[sel])**2)+1e-12)) if np.any(sel) else 0.0

def feats_for_clip(wav: Path):
    x, sr = sf.read(str(wav), always_2d=False)
    if x.ndim>1: x=x.mean(axis=1)
    x = x.astype(np.float32)
    win = max(1,int(sr*0.5)); hop = max(1,int(sr*0.25))
    rows=[]
    for i in range(0,len(x)-win+1,hop):
        fr = x[i:i+win]
        F = np.fft.rfft(fr*np.hamming(len(fr))); mag=np.abs(F)+1e-12; frq=np.fft.rfftfreq(len(fr),1.0/sr)
        tot=float(np.sum(mag))
        centroid=float(np.sum(frq*mag)/tot); flat=float(np.exp(np.mean(np.log(mag)))/(np.mean(mag)+1e-12))
        roll85=float(frq[min(int(np.searchsorted(np.cumsum(mag),0.85*np.sum(mag))),len(frq)-1)])
        peak=float(frq[int(np.argmax(mag))]); tone=float(np.sort(mag)[-5:].sum()/(tot+1e-12))
        low=float(np.sum(mag[(frq>=15)&(frq<80)])/tot); b30=float(np.sum(mag[(frq>=30)&(frq<80)])/tot)
        mid=float(np.sum(mag[(frq>=100)&(frq<300)])/tot); umid=float(np.sum(mag[(frq>=300)&(frq<1000)])/tot)
        high=float(np.sum(mag[(frq>=1000)&(frq<4000)])/tot)
        zcr=float(np.mean(np.sign(fr)[1:]!=np.sign(fr)[:-1]))
        rows.append([float(np.sqrt(np.mean(fr*fr)+1e-12)), zcr, centroid, flat, roll85, peak, tone,
                     low, b30, mid, umid, high, band_rms(fr,sr,15,80), band_rms(fr,sr,100,300), band_rms(fr,sr,1000,4000)])
    return list(np.mean(np.array(rows,dtype=np.float32), axis=0)) if rows else [0.0]*15

def norm_label(s: str):
    L=(s or "").lower()
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
    mfs = sorted(MAN_DIR.glob("manifest_*.csv"))
    if not mfs: print("No manifests. Run auto_ingest.py"); return
    mf = mfs[-1]; print("Using manifest:", mf)
    df = pd.read_csv(mf)
    X=[]; y=[]
    for _,r in df.iterrows():
        seg = Path(str(r["segment"]))
        if not seg.exists(): continue
        lab = str(r.get("manual_label") or r.get("pred_label") or "")
        X.append(feats_for_clip(seg)); y.append(norm_label(lab))
    if not X: print("No data"); return
    X=np.array(X,dtype=np.float32); y=np.array(y)
    classes=sorted(list(set(y))); print("Labels:", classes, "N=", len(y))
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    sc = StandardScaler(); Xtr_s=sc.fit_transform(Xtr)
    clf = LogisticRegression(max_iter=300,class_weight="balanced",multi_class="auto")
    clf.fit(Xtr_s,ytr)
    yhat = clf.predict(sc.transform(Xte))
    print("Report:\n", classification_report(yte,yhat,digits=3))
    print("Confusion matrix:\n", confusion_matrix(yte,yhat,labels=classes))
    (REPO/"models").mkdir(exist_ok=True)
    joblib.dump(clf, REPO/"models/call_model.joblib")
    joblib.dump(sc, REPO/"models/scaler.joblib")
    (REPO/"models/labels.json").write_text(json.dumps(classes))
    print("Saved model to models/")
if __name__=="__main__": main()
