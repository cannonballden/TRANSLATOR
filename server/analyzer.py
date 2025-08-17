import math
from pathlib import Path
from preprocess import extract_audio, extract_frames, load_and_denoise
from audio_models import classify_frames
from vision_models import analyze_frames

def _fuse(audio_segments, visual_cues, fps=1.0):
    max_t = 0.0
    for s in audio_segments: max_t = max(max_t, s["end"])
    for c in visual_cues: max_t = max(max_t, c["time"]+1)
    bins = int(math.ceil(max_t)) if max_t > 0 else 1
    timeline = []
    for i in range(bins):
        t0, t1 = i, i+1
        a_score = {"rumble":0,"trumpet":0,"elephant_vocal":0}
        a_detail = []
        for s in audio_segments:
            overlap = max(0, min(t1, s["end"]) - max(t0, s["start"]))
            if overlap > 0:
                if s["label"] in a_score:
                    a_score[s["label"]] += overlap
                a_detail.append(s["detail"])
        v_score = {"ear_spread":0,"approach":0,"still":0}
        v_detail = []
        for c in visual_cues:
            if t0 <= c["time"] < t1:
                for k in v_score.keys():
                    v_score[k] += 1 if c.get(k) else 0
                v_detail.append(c)
        label, conf, why = "uncertain", 0.2, []
        if a_score["trumpet"] > 0 and (v_score["approach"]>0 or v_score["ear_spread"]>0):
            label, conf = "defensive_trumpet", 0.8
            why = ["audio: trumpet", "vision: approach/ear_spread"]
        elif a_score["rumble"] > 0:
            label, conf = "contact_rumble", 0.7
            why = ["audio: low-frequency rumble"]
        elif v_score["still"] > 0 and sum(a_score.values()) == 0:
            label, conf = "foraging_resting", 0.55
            why = ["vision: still", "audio: none"]
        if v_detail:
            mean_det = sum([d.get("det_conf",0.5) for d in v_detail]) / len(v_detail)
            conf = float(min(0.95, conf * (0.9 + 0.2*mean_det)))
        timeline.append({"start": t0, "end": t1, "label": label,
                         "explanation": {"audio_frames": a_detail[:3],
                                         "vision_frames": v_detail[:3],
                                         "rationale": why},
                         "confidence": round(conf, 2)})
    merged=[]
    for seg in timeline:
        if not merged or seg["label"] != merged[-1]["label"]:
            merged.append(seg)
        else:
            merged[-1]["end"] = seg["end"]
            merged[-1]["confidence"] = round(max(merged[-1]["confidence"], seg["confidence"]), 2)
    return merged

def _summarize(segments):
    cands = [s for s in segments if s["label"]!="uncertain"]
    if not cands:
        return {"text":"Uncertain — no clear elephant signals (noise/occlusion).",
                "confidence":0.2}
    best = max(cands, key=lambda s: s["confidence"])
    m = {
      "defensive_trumpet":"Move away — defensive trumpet with ear-spread/approach detected.",
      "contact_rumble":"Likely contact/call — low-frequency rumble.",
      "foraging_resting":"Likely foraging/resting — slow movement, no high-arousal calls."
    }
    return {"text": m.get(best["label"], "Mixed/uncertain signals."),
            "confidence": best["confidence"]}

def analyze_media(input_path: str):
    work = Path(input_path).parent
    wav = work / (Path(input_path).stem + "_mono16k.wav")
    frames_dir = work / (Path(input_path).stem + "_frames")
    extract_audio(input_path, wav, sr=16000)
    frames_dir = Path(extract_frames(input_path, frames_dir, fps=1.0))
    y, sr = load_and_denoise(wav, target_sr=16000)
    a = classify_frames(y, sr)
    v = analyze_frames(frames_dir, fps=1.0)
    segments = _fuse(a, v, fps=1.0)

    species_conf = 0.0
    if any(s["label"] in ["rumble","trumpet","elephant_vocal"] for s in a):
        species_conf += 0.5
    if any(True for _ in v):
        species_conf += 0.4
    species_conf = round(min(0.95, species_conf), 2)
    summary = _summarize(segments)

    return {
      "species": "African Savanna Elephant (Loxodonta africana)",
      "species_confidence": species_conf,
      "segments": segments,
      "summary": summary,
      "schema_version": "1.0.0"
    }
