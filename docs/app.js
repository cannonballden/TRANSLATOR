const fileInput = document.getElementById('fileInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const statusEl = document.getElementById('status');
const videoEl = document.getElementById('player');
const timeline = document.getElementById('timeline');
const segList = document.getElementById('segments');
const summaryEl = document.getElementById('summary');
const modal = document.getElementById('explainModal');
const explainText = document.getElementById('explainText');
const closeModal = document.getElementById('closeModal');

closeModal.onclick = () => modal.classList.add('hidden');

fileInput.addEventListener('change', () => {
  const f = fileInput.files[0];
  analyzeBtn.disabled = !f;
  if (f && f.type.startsWith('video/')) {
    videoEl.src = URL.createObjectURL(f);
    videoEl.load();
  } else {
    videoEl.removeAttribute('src');
    videoEl.load();
  }
});

function drawTimeline(segments, duration=60) {
  const ctx = timeline.getContext('2d');
  const w = timeline.width = timeline.clientWidth;
  const h = timeline.height = 40;
  ctx.clearRect(0,0,w,h);
  const colors = {
    defensive_trumpet: '#e74c3c',
    contact_rumble: '#3498db',
    foraging_resting: '#27ae60',
    uncertain: '#bdc3c7'
  };
  segments.forEach(s => {
    const x0 = Math.max(0, (s.start/duration)*w);
    const x1 = Math.min(w, (s.end/duration)*w);
    ctx.fillStyle = colors[s.label] || '#aaa';
    ctx.fillRect(x0, 0, Math.max(1, x1-x0), h);
  });
  const legend = document.getElementById('legend');
  legend.innerHTML = Object.entries(colors)
    .map(([k,v]) => `<span style="display:inline-block;width:12px;height:12px;background:${v};vertical-align:middle;margin-right:6px;border-radius:2px"></span>${k}`)
    .join(' &nbsp; ');
}

async function analyzeFile(f) {
  statusEl.textContent = 'Uploading…';
  const form = new FormData();
  form.append('file', f, f.name);
  const SERVER = (localStorage.getItem('serverUrl') || 'http://localhost:8000');
  const resp = await fetch(`${SERVER}/analyze`, { method: 'POST', body: form });
  if (!resp.ok) {
    const msg = await resp.text();
    throw new Error(`Server error: ${resp.status} ${msg}`);
  }
  return await resp.json();
}

function renderResults(res) {
  summaryEl.textContent = `${res.summary.text} (confidence ${(res.summary.confidence*100).toFixed(0)}%)`;
  segList.innerHTML = '';
  const durationGuess = Math.max(...res.segments.map(s => s.end), 10);
  drawTimeline(res.segments, durationGuess);
  res.segments.forEach((s) => {
    const li = document.createElement('li');
    const left = document.createElement('div');
    left.innerHTML = `<span class="seg-label">${s.label}</span>
      <span class="seg-time">${s.start.toFixed(1)}s → ${s.end.toFixed(1)}s</span>`;
    const right = document.createElement('div');
    right.innerHTML = `<span class="seg-conf">${(s.confidence*100).toFixed(0)}%</span>`;
    const btn = document.createElement('button');
    btn.className = 'explain-btn';
    btn.textContent = 'Explain';
    btn.onclick = () => {
      explainText.textContent = JSON.stringify(s.explanation, null, 2);
      modal.classList.remove('hidden');
    };
    right.appendChild(btn);
    li.appendChild(left); li.appendChild(right);
    segList.appendChild(li);
  });
}

analyzeBtn.addEventListener('click', async () => {
  const f = fileInput.files[0];
  if (!f) return;
  analyzeBtn.disabled = true;
  statusEl.textContent = 'Analyzing… (first run may fetch model weights)';
  try {
    const res = await analyzeFile(f);
    renderResults(res);
    statusEl.textContent = `Species: ${res.species}  (p≈${(res.species_confidence*100).toFixed(0)}%)`;
  } catch (e) {
    console.error(e);
    statusEl.textContent = e.message;
  } finally {
    analyzeBtn.disabled = false;
  }
});
