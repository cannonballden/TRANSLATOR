# server/analyzer.py  (NEW FILE)

from pathlib import Path
import time

def analyze_media(filepath: str):
    """
    Minimal stub so /analyze returns a valid JSON and CORS headers flow.
    Replace with real analysis later.
    """
    p = Path(filepath)
    time.sleep(0.3)  # tiny delay to simulate work

    return {
        "file": p.name,
        "species": {"label": "African elephant (stub)", "confidence": 0.82},
        "segments": [
            {
                "start": 0.0,
                "end": 2.5,
                "label": "calling",
                "explanation": "Detected tonal vocal energy (stub).",
                "confidence": 0.76,
                "features": {"audio_bands": [400, 600], "motion": "low"}
            },
            {
                "start": 2.5,
                "end": 6.0,
                "label": "resting",
                "explanation": "Low motion; no high-energy vocalization (stub).",
                "confidence": 0.71,
                "features": {"audio_bands": [], "motion": "very_low"}
            }
        ],
        "summary": "Likely routine contact call; no threat signals (stub).",
        "overall_confidence": 0.74
    }
