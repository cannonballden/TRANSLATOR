(function(){
  const $ = s => document.querySelector(s);
  const dot = $('#dot');
  const serverInput = $('#serverUrl');
  const saveBtn = $('#saveServer');
  const healthBtn = $('#checkHealth');
  const drop = $('#drop');
  const fileInput = $('#fileInput');
  const analyzeBtn = $('#analyzeBtn');
  const downloadBtn = $('#downloadBtn');
  const statusEl = $('#status');
  const prog = $('#prog');
  const filtersPanel = $('#filters');
  const summaryEl = $('#summary');
  const diagEl = $('#diagnostics');
  const timelineEl = $('#timeline');
  const playEl = $('#playbyplay');

  function setDot(state){
    dot.style.background = state === 'ok' ? 'var(--ok)' :
                           state === 'bad' ? 'var(--bad)' : '#666';
  }
  function normUrl(u){ return (u||'').trim().replace(/\/+$/,''); }
  function colorClass(label){
    const L = label.toLowerCase();
    if (L.includes('greeting')) return 'chorus';
    if (L.includes('trumpet')) return 'trumpet';
    if (L.includes('roar')) return 'roar';
    if (L.includes('rumble')) return 'rumble';
    if (L.includes('rest')) return 'resting';
    return 'other';
  }
  function fitsFilter(label, filters){
    const c = colorClass(label);
    if (c==='trumpet') return filters.trumpet;
    if (c==='rumble') return filters.rumble;
    if (c==='roar') return filters.roar;
    if (c==='resting') return filters.resting;
    if (c==='chorus') return filters.chorus;
    return filters.other;
  }

  // persist server URL
  const saved = localStorage.getItem('serverUrl') || '';
  if (saved) serverInput.value = saved;

  async function checkHealth(){
    const u = normUrl(serverInput.value || saved);
    if (!u) { alert('Paste your server URL first.'); return; }
    try{
      const r = await fetch(u+'/health', {mode:'cors'});
      if(!r.ok) throw new Error('HTTP '+r.status);
      const j = await r.json();
      setDot('ok');
      statusEl.textContent = `Server OK (${j.api||'api'})`;
    }catch(e){
      setDot('bad');
      statusEl.textContent = 'Server unreachable';
    }
  }
  saveBtn.addEventListener('click', ()=>{
    const u = normUrl(serverInput.value);
    if (!u) { alert('Paste your server URL first.'); return; }
    localStorage.setItem('serverUrl', u);
    statusEl.textContent = 'Server URL saved.';
  });
  healthBtn.addEventListener('click', checkHealth);

  // drag & drop
  ;['dragenter','dragover'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add('drag');}));
  ;['dragleave','drop'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove('drag');}));
  drop.addEventListener('drop', e=>{
    const f = e.dataTransfer.files?.[0];
    if (f) fileInput.files = e.dataTransfer.files;
  });

  function renderResult(j){
    filtersPanel.classList.remove('hidden');
    // Summary
    summaryEl.textContent = `${j.summary} (overall ${Math.round((j.overall_confidence||0)*100)}%)`;

    // Diagnostics
    const di = j.diagnostics||{};
    diagEl.textContent = `Diagnostics — ffmpeg: ${di.ffmpeg?'yes':'no'}, numpy: ${di.numpy?'yes':'no'}, pillow: ${di.pillow?'yes':'no'}`;

    // Timeline
    const segs = j.segments||[];
    const total = j.duration || (segs.length ? segs[segs.length-1].end : 1) || 1;
    timelineEl.innerHTML = '';

    // Filter state
    const filt = {
      trumpet: document.querySelector('input[data-f="trumpet"]').checked,
      rumble:  document.querySelector('input[data-f="rumble"]').checked,
      roar:    document.querySelector('input[data-f="roar"]').checked,
      resting: document.querySelector('input[data-f="resting"]').checked,
      chorus:  document.querySelector('input[data-f="chorus"]').checked,
      other:   document.querySelector('input[data-f="other"]').checked,
    };

    segs.forEach(seg=>{
      if (!fitsFilter(seg.label, filt)) return;
      const left = 100 * (seg.start / total);
      const width = 100 * ((seg.end - seg.start) / total);
      const d = document.createElement('div');
      const cc = colorClass(seg.label);
      d.className = `seg ${cc}`;
      d.style.left = left+'%';
      d.style.width = Math.max(0.5,width)+'%';
      d.title = `${seg.label} (${seg.start.toFixed(2)}–${seg.end.toFixed(2)}s)`;
      timelineEl.appendChild(d);
    });

    // Play-by-play
    playEl.innerHTML = '';
    segs.forEach(seg=>{
      if (!fitsFilter(seg.label, filt)) return;
      const row = document.createElement('div'); row.className = 'row';
      const t = document.createElement('div'); t.className = 'time';
      t.textContent = `${seg.start.toFixed(2)}–${seg.end.toFixed(2)}s`;
      const body = document.createElement('div');
      const h = document.createElement('div'); h.className = 'label';
      const mv = seg.movement?.level ? `, motion ${seg.movement.level}` : '';
      h.textContent = `${seg.label} — conf ${Math.round((seg.confidence||0)*100)}%${mv}`;
      const exp = document.createElement('details');
      const sum = document.createElement('summary'); sum.textContent = 'Explain';
      const pre = document.createElement('div'); pre.className = 'explain';
      pre.textContent = seg.explanation || JSON.stringify(seg.features || {}, null, 2);
      exp.appendChild(sum); exp.appendChild(pre);
      body.appendChild(h); body.appendChild(exp);
      row.appendChild(t); row.appendChild(body);
      playEl.appendChild(row);
    });

    // Enable download
    const blob = new Blob([JSON.stringify(j,null,2)], {type:'application/json'});
    downloadBtn.href = URL.createObjectURL(blob);
    downloadBtn.download = (j.file || 'analysis') + '.json';
    downloadBtn.disabled = false;
  }

  async function analyze(){
    downloadBtn.disabled = true; downloadBtn.removeAttribute('href');
    const u = normUrl(serverInput.value || localStorage.getItem('serverUrl') || '');
    if (!u) { alert('Set server URL first.'); return; }
    const f = fileInput.files?.[0];
    if (!f) { alert('Choose a clip (audio/video).'); return; }

    analyzeBtn.disabled = true; analyzeBtn.textContent = 'Analyzing…';
    statusEl.textContent = 'Uploading & analyzing…'; prog.classList.remove('hidden'); prog.value = 15;

    try{
      const fd = new FormData();
      fd.append('file', f, f.name);
      prog.value = 35;
      const r = await fetch(u + '/analyze', { method:'POST', body: fd, mode:'cors' });
      prog.value = 70;
      if (!r.ok){
        const txt = await r.text().catch(()=> '');
        throw new Error(`Server error ${r.status}: ${txt}`.slice(0,300));
      }
      const j = await r.json();
      prog.value = 100;
      renderResult(j);
      statusEl.textContent = 'Done.';
    }catch(e){
      console.error(e);
      statusEl.textContent = e.message || 'Failed.';
      alert('Analyze failed: ' + (e.message || e));
    }finally{
      analyzeBtn.disabled = false; analyzeBtn.textContent = 'Analyze';
      setTimeout(()=>prog.classList.add('hidden'), 500);
    }
  }

  // Filters re-render
  document.querySelectorAll('input[data-f]').forEach(cb=>{
    cb.addEventListener('change', ()=>{
      // Force re-render from the last downloaded JSON by clicking Download URL if available:
      if (downloadBtn.href) {
        fetch(downloadBtn.href).then(r=>r.json()).then(renderResult).catch(()=>{});
      }
    });
  });

  analyzeBtn.addEventListener('click', analyze);
  checkHealth(); // silent auto-check if URL already there
})();
