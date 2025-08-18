# QuickFix Pack

This pack makes the Codespace auto-forward port 8000 and auto-start the API.
- Lazy-loads heavy ML on /analyze, so the server comes up instantly.
- Adds .devcontainer with port forwarding and postStart to run Uvicorn.
- Adds VS Code tasks for installing full CPU-only ML deps and running the API manually.

## Apply (in your repo)
1. Upload this zip to your repo (Add file → Upload files) and commit.
2. In Codespaces terminal:
   ```bash
   unzip translate-quickfix.zip
   cp -r translate-quickfix/* .
   rm -rf translate-quickfix translate-quickfix.zip
   git add . && git commit -m "QuickFix: lazy import + devcontainer + tasks" && git push
   ```
3. Ctrl+Shift+P → **Codespaces: Rebuild Container**.
4. After rebuild: open **View → Ports**, set 8000 **Public**; visit /health.
5. In your Pages site console:
   ```js
   localStorage.setItem('serverUrl','<PORT_8000_URL>');
   ```
6. Reload and analyze a clip.
