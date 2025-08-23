#!/usr/bin/env python3
# training/auto_ingest.py
# Downloads 100+ scientifically grounded clips (SoundCloud/Vimeo),
# converts to 16k mono WAV, segments & auto-labels using mini_api,
# and writes a manifest CSV. Parses behavior hints from titles where possible.

import os, sys, json, re, subprocess, shutil
from pathlib import Path
from typing import List
from datetime import datetime
import pandas as pd
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO / "training"
SEG_DIR = TRAIN_DIR / "segments_audio"
MAN_DIR = TRAIN_DIR / "manifests"
SRC = TRAIN_DIR / "sources.yaml"
for d in (SEG_DIR, MAN_DIR): d.mkdir(parents=True, exist_ok=True)

def have(cmd: str) -> bool: return shutil.which(cmd) is not None
def run(cmd: List[str]): return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

def ffmpeg_wav_16k(src: str, dst: str) -> bool:
    if not have("ffmpeg"): print("ffmpeg not found"); return False
    r = run(["ffmpeg","-y","-i",src,"-ac","1","-ar","16000","-vn",dst]); return r.returncode==0

def yt_dlp_audio(url: str, out: Path) -> List[Path]:
    if not have("yt-dlp"): raise RuntimeError("yt-dlp not installed")
    before = set(p.name for p in out.glob("*"))
    tmpl = str(out / "%(title).200B [%(id)s].%(ext)s")
    cmd = ["yt-dlp","-f","bestaudio","-o",tmpl,"--ignore-errors","--no-warnings",
           "--retries","10","--sleep-requests","1","--sleep-interval","1-3", url]
    r = run(cmd)
    after = [p for p in out.glob("*") if p.name not in before]
    return after

def analyze_local(wav_path: Path):
    code = f"""
import json
from pathlib import Path
import mini_api
res = mini_api.analyze_media_file(Path(r\"\"\"{str(wav_path)}\"\"\"), include_motion=False)
print(json.dumps(res))
"""
    r = run([sys.executable,"-c",code])
    if r.returncode!=0: raise RuntimeError("analysis failed: "+r.stdout)
    return json.loads(r.stdout)

def parse_label_from_name(name: str) -> str:
    L = name.lower()
    for k,v in {
        "trumpet":"trumpet","roar":"roar","rumble-roar":"roar",
        "contact rumble":"contact-rumble","contact-rumble":"contact-rumble",
        "greeting":"greeting-chorus","lets-go":"lets-go","let's-go":"lets-go",
        "musth":"musth-like","estrous":"estrous","rumble":"rumble"
    }.items():
        if k in L: return v
    m = re.search(r"ethogram[-\s:]?([a-z\- ]+)", L)
    if m:
        token = m.group(1)
        for k in ("trumpet","roar","contact rumble","greeting","lets-go","musth","estrous","rumble"):
            if k in token: return k.replace(" ","-")
    return ""

def main():
    import argparse, yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=160)
    ap.add_argument("--target_segments", type=int, default=250)
    args = ap.parse_args()

    cfg = yaml.safe_load(SRC.read_text())
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    manifest = MAN_DIR / f"manifest_{stamp}.csv"
    rows = []; work = TRAIN_DIR / "work"; work.mkdir(exist_ok=True)

    # SoundCloud first (rich labels), Vimeo second
    for sc in cfg.get("soundcloud", []):
        try: yt_dlp_audio(sc, work)
        except Exception as e: print("SoundCloud error:", e)

    for v in cfg.get("vimeo_showcases", []):
        try: yt_dlp_audio(v, work)
        except Exception as e: print("Vimeo error:", e)

    media = sorted(work.glob("*.*"))
    if not media:
        print("No media downloaded. Check sources.yaml"); return
    picked = media[:max(100, min(len(media), 320))]
    print("Preparing", len(picked), "items")

    seg_count = 0
    for m in tqdm(picked, desc="Prep+Analyze"):
        wav = m.with_suffix(".wav")
        if not ffmpeg_wav_16k(str(m), str(wav)): continue
        manual_label = parse_label_from_name(m.stem)
        try: res = analyze_local(wav)
        except Exception as e: print("analyze failed:", e); continue
        segs = [s for s in (res.get("segments") or []) if (s.get("end",0)-s.get("start",0))>=0.3]
        segs = sorted(segs, key=lambda s: s.get("confidence",0.0), reverse=True)[:5]
        for s in segs:
            start,end = float(s["start"]), float(s["end"])
            outw = SEG_DIR / (m.stem.replace("/","_")+f"_{int(start*1000)}_{int(end*1000)}.wav")
            r = run(["ffmpeg","-y","-ss",str(start),"-t",str(max(0.1,end-start)),"-i",str(wav),"-ac","1","-ar","16000","-vn",str(outw)])
            if r.returncode==0:
                rows.append({
                    "src": str(m), "segment": str(outw),
                    "start_s": start, "end_s": end,
                    "pred_label": s["label"], "pred_conf": float(s.get("confidence",0.0)),
                    "manual_label": manual_label, "source_license": "see source platform"
                })
                seg_count += 1
        if seg_count >= args.target_segments: break

    pd.DataFrame(rows).to_csv(manifest, index=False)
    print("Wrote manifest:", manifest, "Segments:", len(rows))

if __name__ == "__main__": main()
