(function(){
  const $ = (sel)=>document.querySelector(sel);
  const serverInput = $('#serverUrl');
  const saveBtn = $('#saveServer');
  const healthBtn = $('#checkHealth');
  const healthStatus = $('#healthStatus');
  const fileInput = $('#fileInput');
  const analyzeBtn = $('#analyzeBtn');
  const statusEl = $('#status');
  const results = $('#results');
  const summaryEl = $('#summary');
  const timelineEl = $('#timeline');
  const playbyplayEl = $('#playbyplay');

  // Load persisted server URL
  const saved = localStorage.getItem('serverUrl') || '';
  if (saved) serverInput.value = saved;

  function normUrl(u){
    if(!u) return '';
    return u.replace(/\/+$/,'');
  }

  function setServer(){
    const u = normUrl(serverInput.value.trim());
    if(!u){ alert('Paste your Codespaces server URL first.'); return; }
    localStorage.setItem('serverUrl', u);
    statusEl.textContent = 'Server saved.';
    return u;
  }

  async function checkHealth(){
    const u = normUrl(serverInput.value.trim() || localStorage.getItem('serverUrl') || '');
    if(!u){ alert('Set server URL first.'); return; }
    try{
      healthStatus.textContent = 'checking…';
      const r = await fetch(u+'/health', {mode:'cors'});
      if(!r.ok){ throw new Error('HTTP '+r.status); }
      const j = await r.json();
      healthStatus.textContent = 'ok ('+(j.api || 'api')+')';
      healthStatus.style.color = '#38c172';
    }catch(e){
      healthStatus.textContent = 'unreachable';
      healthStatus.style.color = '#ff5d5d';
    }
  }

  function fmt(s){ return (Math.round(s*100)/100).toFixed(2)+'s'; }
  function colorClass(label){
    if(label.includes('trumpet')) return 'trumpet';
    if(label.includes('rumble')) return 'rumble';
    if(label.includes('growl')) return 'growl';
    if(label.includes('resting')) return 'resting';
    if(label.includes('alarm')) return 'trumpet';
    if(label.includes('contact')) return 'rumble';
    return 'calling';
  }

  function renderResult(j){
    results.classList.remove('hidden');
    summaryEl.textContent = `${j.summary}  (overall ${Math.round((j.overall_confidence||0)*100)}%)`;

    // timeline
    timelineEl.innerHTML = '';
    const total = j.duration || (j.segments?.length ? (j.segments[j.segments.length-1].end||0) : 0) || 1;
    (j.segments||[]).forEach(seg=>{
      const left = (100 * (seg.start/total));
      const width = (100 * ((seg.end-seg.start)/total));
      const el = document.createElement('div');
      el.className = `seg ${colorClass(seg.label)}`;
      el.style.left = left+'%';
      el.style.width = Math.max(1,width)+'%';
      el.title = `${seg.label} (${fmt(seg.start)}–${fmt(seg.end)})`;
      timelineEl.appendChild(el);
    });

    // play-by-play
    playbyplayEl.innerHTML = '';
    (j.segments||[]).forEach((seg, i)=>{
      const row = document.createElement('div');
      row.className = 'row';
      const t = document.createElement('div');
      t.className = 'time';
      t.textContent = `${fmt(seg.start)}–${fmt(seg.end)}`;
      const body = document.createElement('div');
      const h = document.createElement('div');
      h.className = 'label';
      const conf = `, conf ${Math.round((seg.confidence||0)*100)}%`;
      const mv = seg.movement?.level ? `, movement ${seg.movement.level}` : '';
      h.textContent = `${seg.label}${conf}${mv}`;
      const exp = document.createElement('details');
      const sum = document.createElement('summary');
      sum.textContent = 'Explain';
      const pre = document.createElement('div');
      pre.className = 'explain';
      pre.textContent = seg.explanation || JSON.stringify(seg.features||{}, null, 2);
      exp.appendChild(sum);
      exp.appendChild(pre);
      body.appendChild(h);
      body.appendChild(exp);
      row.appendChild(t);
      row.appendChild(body);
      playbyplayEl.appendChild(row);
    });
  }

  async function analyze(){
    statusEl.textContent = '';
    const u = normUrl(serverInput.value.trim() || localStorage.getItem('serverUrl') || '');
    if(!u){ alert('Set server URL first.'); return; }
    const f = fileInput.files?.[0];
    if(!f){ alert('Choose an audio/video file first.'); return; }

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing…';
    statusEl.textContent = 'Uploading & analyzing…';

    try{
      const fd = new FormData();
      fd.append('file', f, f.name);
      const r = await fetch(u+'/analyze', { method:'POST', body: fd, mode:'cors' });
      if(!r.ok){
        const txt = await r.text().catch(()=> '');
        throw new Error(`Server error ${r.status}: ${txt}`);
      }
      const j = await r.json();
      renderResult(j);
      statusEl.textContent = 'Done.';
    }catch(e){
      console.error(e);
      statusEl.textContent = e.message || 'Failed.';
      alert('Analyze failed: '+(e.message||e));
    }finally{
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = 'Analyze';
    }
  }

  saveBtn.addEventListener('click', setServer);
  healthBtn.addEventListener('click', checkHealth);
  analyzeBtn.addEventListener('click', analyze);

  // On load, if server saved, auto health-check (silent)
  if (saved){
    checkHealth();
  }
})();
