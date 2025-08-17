import numpy as np, csv
import tensorflow as tf, tensorflow_hub as hub
import librosa
from preprocess import lowfreq_energy_ratio

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
yamnet = hub.load(YAMNET_URL)

def _load_yamnet_class_map():
    class_map_path = yamnet.class_map_path().numpy().decode("utf-8")
    with tf.io.gfile.GFile(class_map_path, "r") as f:
        rdr = csv.DictReader(f.read().splitlines())
    idx_to_name = {int(row["index"]): row.get("display_name", "") for row in rdr}
    return idx_to_name

IDX_TO_NAME = _load_yamnet_class_map()

def _find_class_indices(substrings):
    out = []
    for idx, name in IDX_TO_NAME.items():
        nm = (name or "").lower()
        if any(s.lower() in nm for s in substrings):
            out.append(idx)
    return out

ELEPHANT_IDXS = _find_class_indices(["elephant"])
ANIMAL_VOCAL_IDXS = _find_class_indices(["animal","roar","growl","squeal"])

def _segment_audio(y, sr, win=0.96, hop=0.48):
    step = int(hop*sr); size = int(win*sr)
    frames = []
    for start in range(0, max(1, len(y)-size), step):
        frames.append((start/sr, (start+size)/sr, y[start:start+size]))
    return frames

def _merge_adjacent(frames, tol=0.2):
    if not frames: return []
    merged = [frames[0].copy()]
    for f in frames[1:]:
        last = merged[-1]
        if f["label"] == last["label"] and abs(f["start"] - last["end"]) <= tol:
            last["end"] = f["end"]
            for k in ["elephant_prob","animal_prob","centroid_hz","lowfreq_ratio"]:
                if k in f["detail"]:
                    last["detail"][k] = max(last["detail"].get(k,0), f["detail"][k])
        else:
            merged.append(f.copy())
    return merged

def classify_frames(y, sr):
    frames = _segment_audio(y, sr)
    out = []
    for (t0,t1, clip) in frames:
        waveform = clip.astype(np.float32)
        scores, embeddings, _ = yamnet(waveform)
        scores = scores.numpy().mean(axis=0)
        ele_p = float(scores[ELEPHANT_IDXS].max()) if ELEPHANT_IDXS else 0.0
        animal_p = float(scores[ANIMAL_VOCAL_IDXS].max()) if ANIMAL_VOCAL_IDXS else 0.0
        sc = float(librosa.feature.spectral_centroid(y=clip, sr=sr).mean())
        lowr = lowfreq_energy_ratio(clip, sr, cutoff=80.0)
        if ele_p > 0.1 or animal_p > 0.1:
            if lowr > 0.55 and sc < 300:
                label = "rumble"
            elif sc > 600:
                label = "trumpet"
            else:
                label = "elephant_vocal"
        else:
            label = "unknown"
        detail = {"elephant_prob": ele_p, "animal_prob": animal_p, "centroid_hz": sc, "lowfreq_ratio": lowr}
        out.append({"start": t0, "end": t1, "label": label, "detail": detail})
    return _merge_adjacent(out)
