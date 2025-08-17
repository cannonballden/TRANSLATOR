# Elephant Translator (Prototype)

A conservative, evidence-based “translator” for **African savanna elephants (Loxodonta africana)**.

Detects **vocalizations** (rumble, trumpet) and visible **body-language cues** (ear-spread proxy, approach, stillness), then outputs a **time-aligned play-by-play** and a concise **summary** with confidences.

## Quickstart (web + cloud; no local installs)
- Use GitHub **Codespaces** to run the API in the browser.
- Serve `/docs/` via **GitHub Pages** as the UI.

### Steps
1. Create a GitHub repo and upload this project (unzip first if needed).
2. Open a **Codespace** → terminal:
   ```bash
   sudo apt-get update && sudo apt-get install -y ffmpeg
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   uvicorn server.app:app --host 0.0.0.0 --port 8000
   ```
   In Ports, make **8000** Public and copy its URL.
3. Visit your GitHub Pages site (Settings → Pages → deploy `/docs`).
4. In the browser console on the Pages site:
   ```js
   localStorage.setItem('serverUrl','<YOUR_PUBLIC_8000_URL>');
   ```
   Reload and upload a sample.

## Local run (optional)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Ethics & limits
Not a language translator—maps cues to behavioral states. Declares uncertainty when evidence is weak.
