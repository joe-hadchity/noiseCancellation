(function() {
  const apiBaseEl = document.getElementById('apiBase');
  const cancelEl = document.getElementById('cancelPct');
  const cancelValueEl = document.getElementById('cancelValue');
  const grantBtn = document.getElementById('grantBtn');
  const startBtn = document.getElementById('startBtn');
  const stopBtn = document.getElementById('stopBtn');
  const statusEl = document.getElementById('status');
  const player = document.getElementById('player');

  let live = false;
  let stream = null;
  let ac = null;
  let source = null;
  let processor = null;
  let buffers = [];
  let total = 0;
  let uploading = false;
  let playCtx = null;
  let playQueueTime = 0;
  const chunkMs = 2000;

  function log(msg) { statusEl.textContent = msg; }

  function floatTo16(input) {
    const out = new Int16Array(input.length);
    for (let i = 0; i < input.length; i++) {
      const s = Math.max(-1, Math.min(1, input[i]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return out;
  }
  function writeString(view, offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }
  function encodeWavMono(samples, sampleRate) {
    const pcm = floatTo16(samples);
    const buffer = new ArrayBuffer(44 + pcm.length * 2);
    const view = new DataView(buffer);
    const bytesPerSample = 2;
    const blockAlign = 1 * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + pcm.length * bytesPerSample, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bytesPerSample * 8, true);
    writeString(view, 36, 'data');
    view.setUint32(40, pcm.length * bytesPerSample, true);
    let offset = 44;
    for (let i = 0; i < pcm.length; i++) {
      view.setInt16(offset, pcm[i], true);
      offset += 2;
    }
    return new Blob([buffer], { type: 'audio/wav' });
  }

  async function uploadChunk(samples, sampleRate) {
    if (uploading) return;
    uploading = true;
    try {
      const blob = encodeWavMono(samples, sampleRate);
      const fd = new FormData();
      fd.append('audio', blob, 'live.wav');
      fd.append('prop_decrease', String(Number(cancelEl.value) / 100));
      fd.append('sample_rate', String(sampleRate));
      const base = apiBaseEl.value || 'http://localhost:8000';
      const res = await fetch(base.replace(/\/$/, '') + '/denoise_chunk', {
        method: 'POST',
        body: fd
      });
      const outBlob = await res.blob();
      const AB = window.AudioContext || window.webkitAudioContext;
      if (!playCtx) playCtx = new AB();
      const buf = await outBlob.arrayBuffer();
      const decoded = await playCtx.decodeAudioData(buf);
      const src = playCtx.createBufferSource();
      src.buffer = decoded;
      src.connect(playCtx.destination);
      const startAt = Math.max(playCtx.currentTime, playQueueTime);
      src.start(startAt);
      playQueueTime = startAt + decoded.duration;
      // also reflect on audio element for visual
      player.src = URL.createObjectURL(outBlob);
    } catch (e) {
      log('Upload error: ' + (e && e.message ? e.message : String(e)));
    } finally {
      uploading = false;
    }
  }

  function concatForLength(chunks, required) {
    let taken = 0;
    const out = new Float32Array(required);
    const rest = [];
    for (let i = 0; i < chunks.length && taken < required; i++) {
      const part = chunks[i];
      const toCopy = Math.min(part.length, required - taken);
      out.set(part.subarray(0, toCopy), taken);
      taken += toCopy;
      if (toCopy < part.length) {
        rest.push(part.subarray(toCopy));
        for (let j = i + 1; j < chunks.length; j++) rest.push(chunks[j]);
        return { take: out, rest };
      }
    }
    return { take: out, rest };
  }

  async function start() {
    if (live) return;
    try {
      const media = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream = media;
      const AB = window.AudioContext || window.webkitAudioContext;
      ac = new AB();
      source = ac.createMediaStreamSource(media);
      processor = ac.createScriptProcessor(4096, 1, 1);
      buffers = [];
      total = 0;
      playQueueTime = 0;
      source.connect(processor);
      const mute = ac.createGain();
      mute.gain.value = 0;
      processor.connect(mute);
      mute.connect(ac.destination);
      const chunkSamples = Math.floor((chunkMs / 1000) * ac.sampleRate);
      processor.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        buffers.push(new Float32Array(input));
        total += input.length;
        if (total >= chunkSamples) {
          const { take, rest } = concatForLength(buffers, chunkSamples);
          buffers = rest;
          total -= chunkSamples;
          uploadChunk(take, ac.sampleRate);
        }
      };
      live = true;
      startBtn.disabled = true;
      stopBtn.disabled = false;
      log('Live started at ' + ac.sampleRate + ' Hz');
    } catch (e) {
      log('Mic error: ' + (e && e.message ? e.message : String(e)));
    }
  }

  function stop() {
    if (!live) return;
    try {
      if (processor) processor.disconnect();
      if (source) source.disconnect();
      if (ac) ac.close();
      if (playCtx) playCtx.close();
      if (stream) for (const t of stream.getTracks()) t.stop();
    } catch {}
    processor = null; source = null; ac = null; stream = null; playCtx = null; playQueueTime = 0;
    buffers = []; total = 0; uploading = false; live = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    log('Stopped');
  }

  async function requestMic() {
    try {
      // Directly trigger the browser's mic prompt via getUserMedia
      const media = await navigator.mediaDevices.getUserMedia({ audio: true });
      for (const t of media.getTracks()) t.stop();
      log('Microphone permission granted');
    } catch (e) {
      log('Mic permission error: ' + (e && e.message ? e.message : String(e)));
    }
  }

  cancelEl.addEventListener('input', () => { cancelValueEl.textContent = cancelEl.value + '%'; });
  grantBtn.addEventListener('click', requestMic);
  startBtn.addEventListener('click', start);
  stopBtn.addEventListener('click', stop);

  chrome.storage.sync.get(['apiBase', 'cancelPct'], (data) => {
    if (data.apiBase) apiBaseEl.value = data.apiBase;
    if (typeof data.cancelPct === 'number') { cancelEl.value = String(data.cancelPct); cancelValueEl.textContent = data.cancelPct + '%'; }
  });
  apiBaseEl.addEventListener('change', () => chrome.storage.sync.set({ apiBase: apiBaseEl.value }));
  cancelEl.addEventListener('change', () => chrome.storage.sync.set({ cancelPct: Number(cancelEl.value) }));
})();


