import urllib.request, pathlib

BASE = pathlib.Path(__file__).resolve().parents[1] / "examples"
BASE.mkdir(parents=True, exist_ok=True)

SAMPLES = [
  ("https://upload.wikimedia.org/wikipedia/commons/1/1d/Elephant_voice_-_trumpeting.ogg",
   "elephant_trumpet.ogg",
   "CC0 – Wikimedia Commons; Elephant voice - trumpeting.ogg"),
  ("https://upload.wikimedia.org/wikipedia/commons/4/43/Elephants_feeding_at_Artis.ogv",
   "elephants_feeding.ogv",
   "CC BY-SA – Wikimedia Commons; Elephants feeding at Artis"),
  ("https://upload.wikimedia.org/wikipedia/commons/8/89/Elephants_in_Hyderabad_Zoo.webm",
   "elephants_hyderabad.webm",
   "CC BY-SA – Wikimedia Commons; Elephants in Hyderabad Zoo"),
  ("https://upload.wikimedia.org/wikipedia/commons/7/74/Infraschall_-_Wie_Elefanten_kommunizieren.webm",
   "infrasound_doc_clip.webm",
   "CC BY-SA – Wikimedia Commons; Infraschall documentary clip")
]

for url, fn, note in SAMPLES:
  dst = BASE / fn
  if not dst.exists():
    print(f"Downloading {fn} …")
    urllib.request.urlretrieve(url, dst)
    print(f"Saved {dst} ({note})")
print("Done.")
