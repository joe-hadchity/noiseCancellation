(function() {
  let enabled = false;
  let apiBase = 'http://localhost:8000';
  let cancelPct = 50;

  chrome.storage.sync.get(['enabled', 'apiBase', 'cancelPct'], (data) => {
    enabled = !!data.enabled;
    if (data.apiBase) apiBase = data.apiBase;
    if (typeof data.cancelPct === 'number') cancelPct = data.cancelPct;
  });
  chrome.storage.onChanged.addListener((changes) => {
    if (changes.enabled) enabled = !!changes.enabled.newValue;
    if (changes.apiBase) apiBase = changes.apiBase.newValue || apiBase;
    if (changes.cancelPct) cancelPct = Number(changes.cancelPct.newValue) || cancelPct;
  });

  const origGetUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);

  // Allow popup to trigger a mic prompt on the meeting origin so the site gets permission
  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg && msg.type === 'REQUEST_MIC') {
      navigator.mediaDevices.getUserMedia({ audio: true }).then((media) => {
        media.getTracks().forEach(t => t.stop());
        sendResponse({ ok: true });
      }).catch((e) => {
        sendResponse({ ok: false, error: e && e.message ? e.message : String(e) });
      });
      return true; // async
    }
  });

  async function denoiseStream(original) {
    // Build a processing graph: capture -> processor -> outStream
    const AC = window.AudioContext || window.webkitAudioContext;
    const ac = new AC();
    const source = ac.createMediaStreamSource(original);
    const dest = ac.createMediaStreamDestination();
    const processor = ac.createScriptProcessor(4096, 1, 1);
    const buffers = [];
    let total = 0;
    const chunkMs = 1000;
    const chunkSamples = Math.floor((chunkMs / 1000) * ac.sampleRate);

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
      for (let i = 0; i < pcm.length; i++) { view.setInt16(offset, pcm[i], true); offset += 2; }
      return new Blob([buffer], { type: 'audio/wav' });
    }

    async function upload(samples, sampleRate) {
      try {
        const fd = new FormData();
        fd.append('audio', encodeWavMono(samples, sampleRate), 'live.wav');
        fd.append('prop_decrease', String(cancelPct / 100));
        fd.append('sample_rate', String(sampleRate));
        const res = await fetch(apiBase.replace(/\/$/, '') + '/denoise_chunk', { method: 'POST', body: fd });
        const blob = await res.blob();
        const buf = await blob.arrayBuffer();
        const decoded = await ac.decodeAudioData(buf);
        const src = ac.createBufferSource();
        src.buffer = decoded;
        src.connect(dest);
        src.start();
      } catch (e) {
        // Swallow errors to avoid breaking mic
      }
    }

    source.connect(processor);
    const mute = ac.createGain(); mute.gain.value = 0; processor.connect(mute); mute.connect(ac.destination);
    processor.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0);
      buffers.push(new Float32Array(input));
      total += input.length;
      if (total >= chunkSamples) {
        let taken = 0; const out = new Float32Array(chunkSamples); const rest = [];
        for (let i = 0; i < buffers.length && taken < chunkSamples; i++) {
          const part = buffers[i]; const toCopy = Math.min(part.length, chunkSamples - taken);
          out.set(part.subarray(0, toCopy), taken); taken += toCopy; if (toCopy < part.length) { rest.push(part.subarray(toCopy)); for (let j = i + 1; j < buffers.length; j++) rest.push(buffers[j]); break; }
        }
        buffers.length = 0; Array.prototype.push.apply(buffers, rest); total -= chunkSamples;
        upload(out, ac.sampleRate);
      }
    };

    return dest.stream;
  }

  navigator.mediaDevices.getUserMedia = async function(constraints) {
    const stream = await origGetUserMedia(constraints);
    if (!enabled) return stream;
    if (constraints && ((constraints.audio && !constraints.video) || constraints.audio)) {
      try {
        return await denoiseStream(stream);
      } catch (e) {
        return stream;
      }
    }
    return stream;
  };
})();


