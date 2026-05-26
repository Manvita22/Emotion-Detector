// UI interactions for Emotion Detector
(() => {
  const modePills = document.querySelectorAll('.mode-pill');
  const speechArea = document.getElementById('speech-area');
  const textArea = document.getElementById('text-area');
  const predictForm = document.getElementById('predict-form');
  const audioInput = document.getElementById('audioFile');
  const player = document.getElementById('player');
  const recordBtn = document.getElementById('record-btn');
  const textInput = document.getElementById('textInput');
  const pasteSample = document.getElementById('paste-sample');
  const textMeta = textArea ? textArea.querySelector('.meta') : null;
  const predictBtn = document.getElementById('predict-btn');
  const resultEl = document.getElementById('result');
  const historyEl = document.getElementById('history');

  let currentMode = 'speech';
  let mediaRecorder, audioChunks = [];

  // Mode switching
  modePills.forEach(p => p.addEventListener('click', () => {
    modePills.forEach(x => x.classList.remove('active'));
    p.classList.add('active');
    currentMode = p.dataset.mode;
    speechArea.classList.toggle('hidden', currentMode !== 'speech');
    textArea.classList.toggle('hidden', currentMode === 'speech');
    if (currentMode === 'fusion') {
      // show both areas
      speechArea.classList.remove('hidden');
      textArea.classList.remove('hidden');
    }
    updateTextPrompt();
  }));

  function updateTextPrompt(){
    if (!textInput) return;
    if (currentMode === 'fusion') {
      textInput.placeholder = 'Optional extra context. Fusion requires audio; ASR transcription is mandatory.';
      pasteSample && pasteSample.classList.add('hidden');
      if (textMeta) textMeta.textContent = 'Fusion requires a successful transcript from the audio';
    } else {
      textInput.placeholder = 'Paste text or drop a .txt file here';
      pasteSample && pasteSample.classList.remove('hidden');
      if (textMeta) textMeta.textContent = 'Max 2000 characters';
    }
  }
  updateTextPrompt();

  // File drag-drop UI
  const dropArea = document.getElementById('drop-area');
  ['dragenter','dragover'].forEach(ev => dropArea.addEventListener(ev,e=>{e.preventDefault();dropArea.classList.add('drag')}));
  ['dragleave','drop'].forEach(ev => dropArea.addEventListener(ev,e=>{e.preventDefault();dropArea.classList.remove('drag')}));
  dropArea.addEventListener('drop', e => {
    const f = e.dataTransfer.files[0];
    if (f) { audioInput.files = e.dataTransfer.files; previewAudio(f); }
  });

  audioInput.addEventListener('change', e => { const f = e.target.files[0]; if (f) previewAudio(f); });
  function previewAudio(file){
    const url = URL.createObjectURL(file); player.src = url; player.classList.remove('hidden');
  }

  // Recording (simple)
  recordBtn.addEventListener('click', async () => {
    if (!mediaRecorder) {
      const stream = await navigator.mediaDevices.getUserMedia({audio:true});
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
      mediaRecorder.onstop = async () => {
        const blob = new Blob(audioChunks,{type:audioChunks[0]?.type||'audio/webm'}); audioChunks = [];
        // Convert recorded blob to 16kHz mono WAV in the browser for reliable server ASR
        try{
          const wavFile = await convertBlobToWavFile(blob, 'recording.wav', 16000);
          const dt = new DataTransfer(); dt.items.add(wavFile); audioInput.files = dt.files; previewAudio(wavFile);
        }catch(err){
          // fallback to original blob file if conversion fails
          const file = new File([blob],'recording.webm',{type:blob.type});
          const dt = new DataTransfer(); dt.items.add(file); audioInput.files = dt.files; previewAudio(file);
        }
      };
    }
    if (mediaRecorder.state === 'recording') { mediaRecorder.stop(); recordBtn.textContent = 'Record'; }
    else { mediaRecorder.start(); recordBtn.textContent = 'Stop'; }
  });

  async function convertBlobToWavFile(blob, filename='recording.wav', targetSampleRate=16000){
    // Decode audio blob to AudioBuffer
    const arrayBuf = await blob.arrayBuffer();
    const ac = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await ac.decodeAudioData(arrayBuf);
    // Get mono data (mix channels)
    const nch = audioBuffer.numberOfChannels;
    const len = audioBuffer.length;
    let pcm;
    if (nch === 1){ pcm = audioBuffer.getChannelData(0); }
    else{
      const tmp = new Float32Array(len);
      for (let c=0;c<nch;c++){ const ch = audioBuffer.getChannelData(c); for (let i=0;i<len;i++) tmp[i]+=ch[i]; }
      for (let i=0;i<len;i++) tmp[i]/=nch;
      pcm = tmp;
    }
    // Resample if needed
    let resampled;
    if (audioBuffer.sampleRate === targetSampleRate){ resampled = pcm; }
    else{ resampled = await resampleFloat32Array(pcm, audioBuffer.sampleRate, targetSampleRate); }
    // Encode WAV PCM16
    const wavBytes = encodeWAV(resampled, targetSampleRate);
    const wavBlob = new Blob([wavBytes], {type: 'audio/wav'});
    return new File([wavBlob], filename, {type: 'audio/wav'});
  }

  function encodeWAV(samples, sampleRate){
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    function writeString(view, offset, string){ for (let i=0;i<string.length;i++){ view.setUint8(offset+i, string.charCodeAt(i)); } }
    /* RIFF identifier */ writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);
    // PCM16 conversion
    let offset = 44;
    for (let i=0;i<samples.length;i++){
      let s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      offset += 2;
    }
    return view;
  }

  // Simple linear resampling (works for short recordings)
  function resampleFloat32Array(buffer, srcRate, dstRate){
    return new Promise(resolve=>{
      const ratio = srcRate / dstRate;
      const dstLength = Math.round(buffer.length / ratio);
      const dst = new Float32Array(dstLength);
      for (let i=0;i<dstLength;i++){
        const srcPos = i * ratio;
        const i0 = Math.floor(srcPos);
        const i1 = Math.min(i0+1, buffer.length-1);
        const frac = srcPos - i0;
        dst[i] = buffer[i0] * (1-frac) + buffer[i1] * frac;
      }
      resolve(dst);
    });
  }

  // Sample insertion for text — prefer embedded window.SAMPLE_LINES (server-side) with fallback to fetch
  let sampleLines = (window.SAMPLE_LINES && Array.isArray(window.SAMPLE_LINES)) ? window.SAMPLE_LINES : null;
  async function ensureSamples(){
    if (sampleLines) return sampleLines;
    try{
      const res = await fetch('/samples/text_only_samples.txt');
      if (!res.ok) return sampleLines = [];
      const txt = await res.text();
      sampleLines = txt.split(/\r?\n/).filter(Boolean);
      return sampleLines;
    }catch(e){ sampleLines = []; return sampleLines; }
  }

  pasteSample && pasteSample.addEventListener('click', async ()=>{
    const lines = await ensureSamples();
    if (!lines || lines.length===0){ textInput.value = 'Could not load samples.'; return; }
    const pick = lines[Math.floor(Math.random()*lines.length)];
    textInput.value = pick; textInput.focus();
  });

  // Submit form
  predictForm.addEventListener('submit', async e => {
    e.preventDefault();
    // UI: loading
    predictBtn.disabled = true; predictBtn.querySelector('.btn-text').textContent = 'Detecting...';
    predictBtn.querySelector('.btn-loader').classList.remove('hidden');
    resultEl.innerHTML = '';

    const form = new FormData();
    form.append('mode', currentMode);
    form.append('model_variant', document.getElementById('modelVariant').value);
    if (audioInput.files[0]) form.append('audio', audioInput.files[0]);
    if (textInput.value && (currentMode !== 'speech')) form.append('text', textInput.value);

    try{
      const res = await fetch('/predict', {method:'POST',body:form});
      const txt = await res.text();
      let body;
      try{ body = JSON.parse(txt); }catch(e){ body = { error: txt || 'Invalid server response' }; }
      renderResult(body);
      addHistory(body);
    }catch(err){
      resultEl.innerHTML = `<div class="placeholder">Failed to predict — ${escapeHtml(String(err))}</div>`;
    }finally{
      predictBtn.disabled = false; predictBtn.querySelector('.btn-text').textContent = 'Detect emotion';
      predictBtn.querySelector('.btn-loader').classList.add('hidden');
    }
  });

  function renderResult(body){
    if (!body || body.error) { resultEl.innerHTML = `<div class="placeholder">${body?.error||'No response'}</div>`; return; }
    const top = body.top || [];
    const note = body.note ? `<div class="result-note">${escapeHtml(body.note)}</div>` : '';
    const transcript = body.transcript ? `
      <div class="result-transcript">
        <strong>Transcript</strong>
        <span>${escapeHtml(body.transcript)}</span>
      </div>` : '';
    resultEl.innerHTML = transcript + note + top.map(item=>{
      const pct = Math.round((item.prob||0)*100);
      return `
        <div class="result-bar">
          <div class="label" title="${escapeHtml(item.label)}">${escapeHtml(item.label)}</div>
          <div class="bar"><div class="bar-fill" style="width:${pct}%"></div></div>
          <div class="score">${pct}%</div>
        </div>`;
    }).join('');
  }

  function addHistory(body){
    const el = document.createElement('div'); el.className='card';
    const title = body.pred?.label || (body.top && body.top[0]?.label) || 'Unknown';
    el.innerHTML = `<div><strong>${escapeHtml(title)}</strong><div class="meta" style="color:var(--muted)">Mode: ${escapeHtml(body.mode||'')}</div></div><div style="color:var(--muted)">${new Date().toLocaleTimeString()}</div>`;
    historyEl.prepend(el);
    // keep history to 8
    while (historyEl.children.length>8) historyEl.removeChild(historyEl.lastChild);
  }

  function escapeHtml(s){ return String(s||'').replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

})();
