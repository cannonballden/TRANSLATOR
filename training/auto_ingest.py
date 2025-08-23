#!/usr/bin/env python3
# training/auto_ingest.py
# Downloads 100+ scientifically grounded clips (SoundCloud/Vimeo/YouTube),
# converts to 16k mono WAV, segments & auto-labels using mini_api,
# and writes a manifest CSV. Tries to parse behavior labels from
# ElephantVoices titles/URLs when possible.

import os, sys, csv, json, re, subprocess, shutil, math, time, random
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import pandas as pd
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO / "training"
SEG_DIR = TRAIN_DIR / "segments_audio"
FEAT_DIR = TRAIN_DIR / "features"
MAN_DIR = TRAIN_DIR / "manifests"
SRC = TRAIN_DIR / "sources.yaml"

for d in (SEG_DIR, FEAT_DIR, MAN_DIR):
    d.mkdir(parents=True, exist_ok=True)

def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

def ffmpeg_wav_16k(src: str, dst: str) -> bool:
    if not have("ffmpeg"):
        print("ffmpeg not found, install it")
        return False
    r = run(["ffmpeg","-y","-i",src,"-ac","1","-ar","16000","-vn",dst])
    return r.returncode == 0

def yt_dlp_audio(url: str, out: Path) -> List[Path]:
    """Download bestaudio for a URL or channel/playlist. Returns list of new files."""
    if not have("yt-dlp"):
        raise RuntimeError("yt-dlp not installed. pip install -r requirements-train.txt")
    out_tmpl = str(out / "%(title).200B [%(id)s].%(ext)s")
    before = set(p.name for p in out.glob("*"))
    cmd = [
        "yt-dlp","-f","bestaudio","-o",out_tmpl,
        "--no-warnings","--ignore-errors","--retry-sleep","2","--retries","10",
        "--no-call-home","--geo-bypass",
        url
    ]
    r = run(cmd)
    if r.returncode != 0:
        print("yt-dlp encountered errors (continuing):\n", r.stdout[:500])
    after = [p for p in out.glob("*") if p.name not in before]
    return after

def list_youtube_items(url: str, limit: int = 160) -> List[str]:
    r = run(["yt-dlp","--flat-playlist","-J",url])
    if r.returncode != 0:
        return []
    try:
        data = json.loads(r.stdout)
    except Exception:
        return []
    entries = data.get("entries", [])
    urls = []
    for e in entries[:limit]:
        vid = e.get("id")
        if vid:
            urls.append("https://www.youtube.com/watch?v=" + vid)
    return urls

def s3_sample(bucket: str, site: str, max_files: int = 3) -> List[str]:
    if not have("aws"):
        print("aws cli not found; skipping ELP Congo sampling")
        return []
    base = f"{bucket}/{site}/"
    r = run(["aws","s3","ls", base, "--no-sign-request"])
    if r.returncode != 0:
        print("aws s3 ls failed; skipping site", site)
        return []
    files = []
    for line in r.stdout.splitlines():
        parts = line.strip().split()
        if parts and parts[-1].endswith(".wav"):
            files.append(base + parts[-1])
    return files[:max_files]

def ffmpeg_cut(src: str, dst: str, start: float, dur: float) -> bool:
    r = run(["ffmpeg","-y","-ss",str(start),"-t",str(dur),"-i",src,"-ac","1","-ar","16000","-vn",dst])
    return r.returncode == 0

def analyze_local(wav_path: Path) -> Dict[str, Any]:
    """Call mini_api.analyze_media_file directly (no HTTP) to get segments.
       IMPORTANT: run this script from the REPO ROOT so 'import mini_api' works."""
    code = f"""
import json
from pathlib import Path
import mini_api
res = mini_api.analyze_media_file(Path(r\"\"\"{str(wav_path)}\"\"\"), include_motion=False)
print(json.dumps(res))
"""
    r = run([sys.executable,"-c",code])
    if r.returncode != 0:
        raise RuntimeError("analysis failed: " + r.stdout)
    return json.loads(r.stdout)

# ----- label parsing from title/URL -----
KNOWN = {
    "trumpet": "trumpet",
    "trumpet-blast": "trumpet",
    "roar": "roar",
    "rumble-roar": "roar",
    "contact rumble": "contact-rumble",
    "contact-rumble": "contact-rumble",
    "greeting rumble": "greeting-chorus",
    "greeting-rumble": "greeting-chorus",
    "let's-go": "lets-go",
    "lets-go": "lets-go",
    "musth": "musth-like",
    "estrous": "estrous",
    "musth-like": "musth-like",
    "rumble": "rumble"
}

def parse_label_from_name(name: str) -> str:
    L = name.lower()
    m = re.search(r"ethogram[-\s:]?([a-z\- ]+)", L)
    if m:
        token = m.group(1).strip()
        for k,v in KNOWN.items():
            if k in token:
                return v
    for k,v in KNOWN.items():
        if k in L:
            return v
    return ""

def load_yaml():
    import yaml
    return yaml.safe_load(SRC.read_text())

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=160, help="max items per source list")
    ap.add_argument("--target_segments", type=int, default=250, help="aim for at least this many segments")
    args = ap.parse_args()

    cfg = load_yaml()
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    manifest = MAN_DIR / f"manifest_{stamp}.csv"

    rows = []
    work = TRAIN_DIR / "work"
    work.mkdir(exist_ok=True)

    # 1) SoundCloud (ElephantVoices)
    sc_urls = cfg.get("soundcloud", [])
    for sc in sc_urls:
        try:
            print("SoundCloud:", sc)
            _ = yt_dlp_audio(sc, work)
        except Exception as e:
            print("SoundCloud failed:", e)

    # 2) YouTube (playlists or channels)
    yt_all = []
    for y in cfg.get("youtube_playlists", []):
        items = list_youtube_items(y, limit=args.limit)
        yt_all.extend(items)
    print("YouTube items:", len(yt_all))
    for u in tqdm(yt_all, desc="YouTube downloads"):
        try:
            _ = yt_dlp_audio(u, work)
        except Exception as e:
            print("YouTube item failed:", e)

    # 3) Vimeo showcase
    for v in cfg.get("vimeo_showcases", []):
        try:
            _ = yt_dlp_audio(v, work)
        except Exception as e:
            print("Vimeo failed:", e)

    # Convert to WAV and analyze
    media = sorted(work.glob("*.*"))
    print("Found media items:", len(media))
    if not media:
        print("No media downloaded. Check sources.yaml.")
        return

    picked = media[:max(100, min(len(media), 320))]
    print("Preparing", len(picked), "items -> WAV + segmenting")

    seg_count = 0
    for m in tqdm(picked, desc="Prep+Analyze"):
        wav = m.with_suffix(".wav")
        if not ffmpeg_wav_16k(str(m), str(wav)):
            continue

        manual_label = parse_label_from_name(m.stem)

        try:
            res = analyze_local(wav)
        except Exception as e:
            print("analyze failed:", e); continue

        segs = res.get("segments", [])
        segs = [s for s in segs if s.get("end",0)-s.get("start",0) >= 0.3]
        segs = sorted(segs, key=lambda s: s.get("confidence",0.0), reverse=True)[:5]

        for s in segs:
            start, end = float(s["start"]), float(s["end"])
            pred_label = s["label"]
            outw = SEG_DIR / (m.stem.replace("/", "_") + f"_{int(start*1000)}_{int(end*1000)}.wav")
            if ffmpeg_cut(str(wav), str(outw), start, max(0.1, end-start)):
                rows.append({
                    "src": str(m),
                    "segment": str(outw),
                    "start_s": start,
                    "end_s": end,
                    "pred_label": pred_label,
                    "pred_conf": float(s.get("confidence",0.0)),
                    "manual_label": manual_label,
                    "source_license": "see source platform page",
                    "url_hint": ""
                })
                seg_count += 1

        if seg_count >= args.target_segments:
            break

    mf = manifest
    pd.DataFrame(rows).to_csv(mf, index=False)
    print("Wrote manifest:", mf)
    print("Segments:", len(rows))
    print("Done.")

if __name__ == "__main__":
    main()
