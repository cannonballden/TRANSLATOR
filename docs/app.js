(function(){
  const $ = s => document.querySelector(s);

  const dot = $('#dot');
  const serverInput = $('#serverUrl');
  const saveBtn = $('#saveServer');
  const healthBtn = $('#checkHealth');
  const configLine = $('#configLine');

  const drop = $('#drop');
  const fileInput = $('#fileInput');
  const motionToggle = $('#motionToggle');
  const analyzeBtn = $('#analyzeBtn');
  const downloadBtn = $('#downloadBtn');
  const statusEl = $('#status');
  const prog = $('#prog');

  const playerWrap = $('#playerWrap');
  const videoEl = $('#player');
  const audioEl = $('#audioPlayer');

  const filtersPanel = $('#filters');
  const summaryEl = $('#summary');
  const diagEl = $('#diagnostics');
  const timelineEl = $('#timeline');
  const markersEl = $('#markers');
  const playEl = $('#playbyplay');

  let cfg = { small_upload_max_mb: 50, max_mb: 1024, chunk_mb: 5 };
  let lastResult = null;
  let objectUrl = null;

  function setDot(state){
    dot.style.background = state === 'ok' ? 'var(--ok)' :
                           state === 'bad' ? 'var(--bad)' : '#666';
  }
  function normUrl(u){ return (u||'').trim().replace(/\/+$/,''); }
  function colorClass(label){
    const L = (label||'').toLowerCase();
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
  function isVideo(file){
    return file && file.type && file.type.startsWith('video');
  }

  // persist server URL
  const saved = localStorage.getItem('serverUrl') || '';
  if (saved) serverInput.value = saved;

  async function getConfig(){
    const u = normUrl(serverInput.value || saved);
    if (!u) return;
    try{
      const r = await fetch(u+'/config', {mode:'cors'});
      if (!r.ok) return;
      cfg = await r.json();
      configLine.textContent = `Server config: small=${cfg.small_upload_max_mb}MB, max=${cfg.max_mb}MB, chunk=${cfg.chunk_mb}MB`;
    }catch{}
  }

  async function checkHealth(){
    const u = normUrl(serverInput.value || saved);
    if (!u) { alert('Paste your server URL first.'); return; }
    try{
      const r = await fetch(u+'/health', {mode:'cors'});
      if(!r.ok) throw new Error('HTTP '+r.status);
      const j = await r.json();
      setDot('ok');
      statusEl.textContent = `Server OK (${j.api||'api'})`;
      await getConfig();
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
    getConfig();
  });
  healthBtn.addEventListener('click', checkHealth);

  // Drag & drop
  ;['dragenter','dragover'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add('drag');}));
  ;['dragleave','drop'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove('drag');}));
  drop.addEventListener('drop', e=>{
    const f = e.dataTransfer.files?.[0];
    if (f) { fileInput.files = e.dataTransfer.files; previewFile(f); }
  });
  fileInput.addEventListener('change', ()=>{
    const f = fileInput.files?.[0];
    if (f) previewFile(f);
  });

  function previewFile(f){
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(f);
    playerWrap.classList.remove('hidden');
    if (isVideo(f)){
      videoEl.src = objectUrl; videoEl.classList.remove('hidden');
      audioEl.src = ''; audioEl.classList.add('hidden');
    } else {
      audioEl.src = objectUrl; audioEl.classList.remove('hidden');
      videoEl.src = ''; videoEl.classList.add('hidden');
    }
  }

  function renderMarkers(total, events){
    markersEl.innerHTML = '';
    if (!events || !events.length) return;
    events.forEach(ev=>{
      if (ev.t == null) return; // global events
      const left = Math.max(0, Math.min(100, 100 * (ev.t / total)));
      const m = document.createElement('div');
      m.className = 'marker';
      const t = (ev.type || '').toLowerCase();
      if (t.includes('freeze')) m.classList.add('freeze');
      else if (t.includes('stomp')) m.classList.add('stomp');
      else if (t.includes('ear')) m.classList.add('ear');
      else if (t.includes('coherent')) m.classList.add('coherent');
      else if (t.includes('excitement')) m.classList.add('excite');
      m.style.left = left + '%';
      m.title = ev.type + (ev.detail ? ` — ${ev.detail}` : '');
      markersEl.appendChild(m);
    });
  }

  function renderResult(j){
    lastResult = j;
    filtersPanel.classList.remove('hidden');
    summaryEl.textContent = `${j.summary} (overall ${Math.round((j.overall_confidence||0)*100)}%)`;
    const di = j.diagnostics||{};
    const extra = (j.findings && j.findings.length) ? `; extra: ${j.findings.join(', ')}` : '';
    diagEl.textContent = `Diagnostics — ffmpeg: ${di.ffmpeg?'yes':'no'}, numpy: ${di.numpy?'yes':'no'}, pillow: ${di.pillow?'yes':'no'}, opencv: ${di.opencv?'yes':'no'}${extra}`;

    const segs = j.segments||[];
    const total = j.duration || (segs.length ? segs[segs.length-1].end : 1) || 1;

    // timeline segments
    timelineEl.innerHTML = '';
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

    // motion markers
    renderMarkers(total, j.motion_events || []);

    // click-to-seek on timeline
    timelineEl.onclick = (e)=>{
      const rect = timelineEl.getBoundingClientRect();
      const frac = Math.max(0, Math.min(1, (e.clientX - rect.left)/rect.width));
      const t = total * frac;
      if (!playerWrap.classList.contains('hidden')){
        if (!videoEl.classList.contains('hidden')) { videoEl.currentTime = t; videoEl.play(); }
        else if (!audioEl.classList.contains('hidden')) { audioEl.currentTime = t; audioEl.play(); }
      }
    };

    // play-by-play
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
      // click row to seek to segment start
      row.onclick = ()=> {
        const start = seg.start || 0;
        if (!playerWrap.classList.contains('hidden')){
          if (!videoEl.classList.contains('hidden')) { videoEl.currentTime = start; videoEl.play(); }
          else if (!audioEl.classList.contains('hidden')) { audioEl.currentTime = start; audioEl.play(); }
        }
      };
      playEl.appendChild(row);
    });

    // enable download
    const blob = new Blob([JSON.stringify(j,null,2)], {type:'application/json'});
    downloadBtn.href = URL.createObjectURL(blob);
    downloadBtn.download = (j.file || 'analysis') + '.json';
    downloadBtn.disabled = false;
  }

  // Re-render when filters change (from lastResult)
  document.querySelectorAll('input[data-f]').forEach(cb=>{
    cb.addEventListener('change', ()=>{
      if (lastResult) renderResult(lastResult);
    });
  });

  async function analyze(){
    downloadBtn.disabled = true; downloadBtn.removeAttribute('href');

    const u = normUrl(serverInput.value || localStorage.getItem('serverUrl') || '');
    if (!u) { alert('Set server URL first.'); return; }
    const f = fileInput.files?.[0];
    if (!f) { alert('Choose a clip (audio/video).'); return; }

    const includeMotion = !!motionToggle.checked;
    const mb = Math.ceil(f.size/1024/1024);
    const useChunked = mb > (cfg.small_upload_max_mb || 50);

    analyzeBtn.disabled = true; analyzeBtn.textContent = useChunked ? 'Uploading (large)…' : 'Analyzing…';
    statusEl.textContent = useChunked ? `Large file mode: ${mb} MB` : 'Uploading & analyzing…';
    prog.classList.remove('hidden'); prog.value = 5;

    try{
      if (!useChunked){
        const fd = new FormData();
        fd.append('file', f, f.name);
        const r = await fetch(u + '/analyze?include_motion=' + String(includeMotion), { method:'POST', body: fd, mode:'cors' });
        if (!r.ok){ const txt = await r.text().catch(()=> ''); throw new Error(`Server error ${r.status}: ${txt}`.slice(0,300)); }
        prog.value = 90;
        const j = await r.json(); prog.value = 100; renderResult(j);
        statusEl.textContent = 'Done.';
      } else {
        // chunked
        const initR = await fetch(u+'/upload/init', {
          method:'POST', mode:'cors',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({ filename:f.name, filesize:f.size, mime:f.type })
        });
        if (!initR.ok){ const t= await initR.text().catch(()=> ''); throw new Error(`Init failed: ${t}`.slice(0,300)); }
        const initJ = await initR.json();
        const uploadId = initJ.upload_id; const chunkBytes = initJ.chunk_bytes || (cfg.chunk_mb*1024*1024) || (5*1024*1024);

        const totalChunks = Math.ceil(f.size / chunkBytes);
        for (let i=0;i<totalChunks;i++){
          const start = i*chunkBytes; const end = Math.min(f.size, (i+1)*chunkBytes);
          const blob = f.slice(start, end);
          const partR = await fetch(u+`/upload/part?upload_id=${uploadId}&index=${i}`, {
            method:'POST', mode:'cors', body: blob
          });
          if (!partR.ok){ const t= await partR.text().catch(()=> ''); throw new Error(`Chunk ${i} failed: ${t}`.slice(0,300)); }
          prog.value = Math.round( ( (i+1)/totalChunks ) * 80 );
          statusEl.textContent = `Uploaded ${i+1}/${totalChunks} chunks…`;
        }

        analyzeBtn.textContent = 'Analyzing…'; statusEl.textContent = 'Analyzing on server…';
        const finR = await fetch(u+`/upload/finish?upload_id=${uploadId}&include_motion=${String(includeMotion)}&total_chunks=${totalChunks}`, {
          method:'POST', mode:'cors'
        });
        if (!finR.ok){ const t= await finR.text().catch(()=> ''); throw new Error(`Finish failed: ${t}`.slice(0,300)); }
        prog.value = 95;
        const j = await finR.json(); prog.value = 100; renderResult(j);
        statusEl.textContent = 'Done.';
      }
    }catch(e){
      console.error(e);
      statusEl.textContent = e.message || 'Failed.';
      alert('Analyze failed: ' + (e.message || e));
    }finally{
      analyzeBtn.disabled = false; analyzeBtn.textContent = 'Analyze';
      setTimeout(()=>prog.classList.add('hidden'), 700);
    }
  }

  analyzeBtn.addEventListener('click', analyze);

  // On load: auto health/config if server already set
  const savedUrl = localStorage.getItem('serverUrl');
  if (savedUrl){ (async ()=>{ await checkHealth(); })(); }
})();
