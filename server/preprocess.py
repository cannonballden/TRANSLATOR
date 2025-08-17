import subprocess, shlex
from pathlib import Path
import numpy as np
import librosa
import noisereduce as nr

FFMPEG = "ffmpeg"

def run(cmd: str):
    p = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", errors="ignore"))
    return p

def extract_audio(input_path: str, out_wav: Path, sr: int = 16000):
    out = Path(out_wav)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = f'{FFMPEG} -y -i \"{input_path}\" -ac 1 -ar {sr} -vn \"{out}\"'
    run(cmd)
    return str(out)

def extract_frames(input_path: str, out_dir: Path, fps: float = 1.0):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cmd = f'{FFMPEG} -y -i \"{input_path}\" -vf \"fps={fps}\" \"{out}/frame_%06d.jpg\"'
    run(cmd)
    return str(out)

def load_and_denoise(wav_path: str, target_sr: int = 16000):
    y, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    n = int(min(len(y), target_sr*0.5))
    noise_clip = y[:n] if n > 0 else y[:1000]
    y_dn = nr.reduce_noise(y=y, y_noise=noise_clip, sr=target_sr)
    y_dn = y_dn / (np.max(np.abs(y_dn)) + 1e-8)
    return y_dn, target_sr

def lowfreq_energy_ratio(y, sr, cutoff=80.0):
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low = S[(freqs <= cutoff)].sum()
    tot = S.sum() + 1e-9
    return float(low / tot)
