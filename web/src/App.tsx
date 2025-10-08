import React, { useEffect, useMemo, useRef, useState } from 'react'
import {
  AppBar,
  Box,
  Button,
  Card,
  CardContent,
  Container,
  Divider,
  IconButton,
  Paper,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Slider,
  LinearProgress,
  Stack,
  Toolbar,
  Typography,
  Snackbar,
  Alert,
} from '@mui/material'
import UploadIcon from '@mui/icons-material/UploadFile'
import GraphicEqIcon from '@mui/icons-material/GraphicEq'
import CleaningServicesIcon from '@mui/icons-material/CleaningServices'
import DownloadIcon from '@mui/icons-material/Download'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import axios from 'axios'

// Removed wavesurfer import to avoid bundling issues; using Canvas rendering instead

const API_BASE = (import.meta as any).env?.VITE_API_BASE ?? 'http://localhost:8000'

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [predicting, setPredicting] = useState(false)
  const [denoising, setDenoising] = useState(false)
  const [probs, setProbs] = useState<number[] | null>(null)
  const [labels, setLabels] = useState<string[] | null>(null)
  const [topIdx, setTopIdx] = useState<number[] | null>(null)
  const [cleanUrl, setCleanUrl] = useState<string | null>(null)
  const [rawUrl, setRawUrl] = useState<string | null>(null)
  const [cancelPct, setCancelPct] = useState<number>(50)
  const [dragOver, setDragOver] = useState<boolean>(false)
  const [snackOpen, setSnackOpen] = useState<boolean>(false)
  const [snackMsg, setSnackMsg] = useState<string>('')
  const [snackSev, setSnackSev] = useState<'success' | 'error' | 'info'>('info')
  const audioRawRef = useRef<HTMLAudioElement>(null)
  const audioCleanRef = useRef<HTMLAudioElement>(null)
  const rawCanvasRef = useRef<HTMLCanvasElement>(null)
  const cleanCanvasRef = useRef<HTMLCanvasElement>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const liveAudioCtxRef = useRef<AudioContext | null>(null)
  const livePlayCtxRef = useRef<AudioContext | null>(null)
  const livePlayQueueTimeRef = useRef<number>(0)
  const liveStreamRef = useRef<MediaStream | null>(null)
  const liveSourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const liveProcessorRef = useRef<ScriptProcessorNode | null>(null)
  const liveBuffersRef = useRef<Float32Array[]>([])
  const liveTotalSamplesRef = useRef<number>(0)
  const liveUploadingRef = useRef<boolean>(false)
  const [liveOn, setLiveOn] = useState<boolean>(false)
  const [liveChunkMs, setLiveChunkMs] = useState<number>(2000)

  const enableActions = !!file && !predicting && !denoising
  const disableFileActions = liveOn || predicting || denoising

  const onPick: React.ChangeEventHandler<HTMLInputElement> = (e) => {
    const f = e.target.files?.[0]
    if (f) {
      setFile(f)
      setProbs(null)
      setLabels(null)
      setTopIdx(null)
      setCleanUrl(null)
      const url = URL.createObjectURL(f)
      setRawUrl(url)
    }
  }

  const onDropFile: React.DragEventHandler<HTMLDivElement> = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragOver(false)
    const f = e.dataTransfer.files?.[0]
    if (f) {
      setFile(f)
      setProbs(null)
      setLabels(null)
      setTopIdx(null)
      setCleanUrl(null)
      const url = URL.createObjectURL(f)
      setRawUrl(url)
      setSnackMsg('File loaded')
      setSnackSev('success')
      setSnackOpen(true)
    }
  }

  const onDragOver: React.DragEventHandler<HTMLDivElement> = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (!dragOver) setDragOver(true)
  }
  const onDragLeave: React.DragEventHandler<HTMLDivElement> = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (dragOver) setDragOver(false)
  }

  const doPredict = async () => {
    if (!file) return
    setPredicting(true)
    try {
      const fd = new FormData()
      fd.append('audio', file)
      const res = await axios.post(`${API_BASE}/predict`, fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setProbs(res.data.probabilities)
      setLabels(res.data.top_labels)
      setTopIdx(res.data.top_indices)
      setSnackMsg('Prediction complete')
      setSnackSev('success')
      setSnackOpen(true)
    } catch (err: any) {
      setSnackMsg(err?.message || 'Prediction failed')
      setSnackSev('error')
      setSnackOpen(true)
    } finally {
      setPredicting(false)
    }
  }

  const doDenoise = async () => {
    if (!file) return
    setDenoising(true)
    try {
      const fd = new FormData()
      fd.append('audio', file)
      fd.append('prop_decrease', String(cancelPct / 100))
      const res = await axios.post(`${API_BASE}/denoise_chunk`, fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
        responseType: 'blob',
      })
      const blob = new Blob([res.data], { type: 'audio/wav' })
      const url = URL.createObjectURL(blob)
      setCleanUrl(url)
      setSnackMsg('Denoised file is ready')
      setSnackSev('success')
      setSnackOpen(true)
    } catch (err: any) {
      setSnackMsg(err?.message || 'Denoise failed')
      setSnackSev('error')
      setSnackOpen(true)
    } finally {
      setDenoising(false)
    }
  }

  // ---- Live denoise helpers ----
  function floatTo16BitPCM(input: Float32Array): Int16Array {
    const out = new Int16Array(input.length)
    for (let i = 0; i < input.length; i++) {
      const s = Math.max(-1, Math.min(1, input[i]))
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff
    }
    return out
  }

  function writeString(view: DataView, offset: number, str: string) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i))
  }

  function writeWavHeader(view: DataView, sampleRate: number, numSamples: number, numChannels: number) {
    const bytesPerSample = 2
    const blockAlign = numChannels * bytesPerSample
    const byteRate = sampleRate * blockAlign
    // ChunkID 'RIFF'
    writeString(view, 0, 'RIFF')
    // ChunkSize = 36 + Subchunk2Size
    view.setUint32(4, 36 + numSamples * bytesPerSample, true)
    // Format 'WAVE'
    writeString(view, 8, 'WAVE')
    // Subchunk1ID 'fmt '
    writeString(view, 12, 'fmt ')
    // Subchunk1Size 16 for PCM
    view.setUint32(16, 16, true)
    // AudioFormat PCM=1
    view.setUint16(20, 1, true)
    // NumChannels
    view.setUint16(22, numChannels, true)
    // SampleRate
    view.setUint32(24, sampleRate, true)
    // ByteRate
    view.setUint32(28, byteRate, true)
    // BlockAlign
    view.setUint16(32, blockAlign, true)
    // BitsPerSample
    view.setUint16(34, bytesPerSample * 8, true)
    // Subchunk2ID 'data'
    writeString(view, 36, 'data')
    // Subchunk2Size
    view.setUint32(40, numSamples * bytesPerSample, true)
  }

  function encodeWavMono(samples: Float32Array, sampleRate: number): Blob {
    const pcm = floatTo16BitPCM(samples)
    const buffer = new ArrayBuffer(44 + pcm.length * 2)
    const view = new DataView(buffer)
    writeWavHeader(view, sampleRate, pcm.length, 1)
    let offset = 44
    for (let i = 0; i < pcm.length; i++) {
      view.setInt16(offset, pcm[i], true)
      offset += 2
    }
    return new Blob([buffer], { type: 'audio/wav' })
  }

  async function uploadLiveChunk(samples: Float32Array, sampleRate: number) {
    if (liveUploadingRef.current) return
    liveUploadingRef.current = true
    try {
      console.log('[live] sending chunk', { samples: samples.length, sampleRate })
      const blob = encodeWavMono(samples, sampleRate)
      const fd = new FormData()
      fd.append('audio', blob, 'live.wav')
      fd.append('prop_decrease', String(cancelPct / 100))
      fd.append('sample_rate', String(sampleRate))
      const res = await axios.post(`${API_BASE}/denoise_chunk`, fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
        responseType: 'blob',
        timeout: 15000,
      })
      console.log('[live] received chunk response', { size: (res.data as any)?.size ?? 'n/a' })
      const outBlob = new Blob([res.data], { type: 'audio/wav' })
      const url = URL.createObjectURL(outBlob)
      setCleanUrl(url)
      // Programmatic playback via WebAudio for reliable output
      const AC = (window as any).AudioContext || (window as any).webkitAudioContext
      const ctx: AudioContext = livePlayCtxRef.current ?? new AC()
      livePlayCtxRef.current = ctx
      const buf = await outBlob.arrayBuffer()
      try {
        if (ctx.state === 'suspended') {
          try { await ctx.resume() } catch {}
        }
        const decoded = await ctx.decodeAudioData(buf)
        const src = ctx.createBufferSource()
        src.buffer = decoded
        src.connect(ctx.destination)
        const startAt = Math.max(ctx.currentTime, livePlayQueueTimeRef.current)
        src.start(startAt)
        livePlayQueueTimeRef.current = startAt + decoded.duration
      } catch (e) {
        // Fallback: try assigning to audio element if decode fails
        const el = audioCleanRef.current
        if (el) {
          try {
            const url2 = URL.createObjectURL(new Blob([buf], { type: 'audio/wav' }))
            el.src = url2
            await el.play()
          } catch {}
        }
        setSnackMsg('Playback error: trying fallback output')
        setSnackSev('info')
        setSnackOpen(true)
      }
      // Notify on first chunk
      if (!snackOpen) {
        setSnackMsg('Receiving live denoised audio...')
        setSnackSev('success')
        setSnackOpen(true)
      }
    } catch (err: any) {
      console.error('[live] upload error', err)
      const msg = err?.response?.data?.detail || err?.message || 'Live denoise failed'
      setSnackMsg(String(msg))
      setSnackSev('error')
      setSnackOpen(true)
    } finally {
      liveUploadingRef.current = false
    }
  }

  function concatForLength(chunks: Float32Array[], required: number): { take: Float32Array, rest: Float32Array[] } {
    let taken = 0
    const out = new Float32Array(required)
    const rest: Float32Array[] = []
    for (let i = 0; i < chunks.length && taken < required; i++) {
      const part = chunks[i]
      const toCopy = Math.min(part.length, required - taken)
      out.set(part.subarray(0, toCopy), taken)
      taken += toCopy
      if (toCopy < part.length) {
        // leftover
        rest.push(part.subarray(toCopy))
        for (let j = i + 1; j < chunks.length; j++) rest.push(chunks[j])
        return { take: out, rest }
      }
    }
    // used all chunks exactly
    return { take: out, rest }
  }

  async function startLive() {
    if (liveOn) return
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      liveStreamRef.current = stream
      const AC = (window as any).AudioContext || (window as any).webkitAudioContext
      const ac: AudioContext = new AC()
      liveAudioCtxRef.current = ac
      const source = ac.createMediaStreamSource(stream)
      liveSourceRef.current = source
      const processor = ac.createScriptProcessor(4096, 1, 1)
      liveProcessorRef.current = processor
      liveBuffersRef.current = []
      liveTotalSamplesRef.current = 0
      source.connect(processor)
      // Connect through a muted gain node so onaudioprocess fires without audible echo
      const mute = ac.createGain()
      mute.gain.value = 0
      processor.connect(mute)
      mute.connect(ac.destination)
      const chunkSamples = Math.floor((liveChunkMs / 1000) * ac.sampleRate)
      console.log('[live] started', { sampleRate: ac.sampleRate, chunkSamples })
      processor.onaudioprocess = async (e) => {
        const input = e.inputBuffer.getChannelData(0)
        // Log once in a while to confirm it's firing
        if (Math.random() < 0.02) {
          console.log('[live] onaudioprocess', { frame: input.length })
        }
        liveBuffersRef.current.push(new Float32Array(input))
        liveTotalSamplesRef.current += input.length
        if (liveTotalSamplesRef.current >= chunkSamples) {
          const { take, rest } = concatForLength(liveBuffersRef.current, chunkSamples)
          liveBuffersRef.current = rest
          liveTotalSamplesRef.current -= chunkSamples
          uploadLiveChunk(take, ac.sampleRate)
        }
      }
      setLiveOn(true)
      setSnackMsg('Live denoise started')
      setSnackSev('success')
      setSnackOpen(true)
    } catch (err: any) {
      console.error('[live] start error', err)
      setSnackMsg(err?.message || 'Microphone permission denied')
      setSnackSev('error')
      setSnackOpen(true)
    }
  }

  function stopLive() {
    setLiveOn(false)
    try {
      liveProcessorRef.current?.disconnect()
      liveSourceRef.current?.disconnect()
      liveAudioCtxRef.current?.close()
      livePlayCtxRef.current?.close()
    } catch {}
    liveProcessorRef.current = null
    liveSourceRef.current = null
    liveAudioCtxRef.current = null
    livePlayCtxRef.current = null
    livePlayQueueTimeRef.current = 0
    liveBuffersRef.current = []
    liveTotalSamplesRef.current = 0
    if (liveStreamRef.current) {
      for (const t of liveStreamRef.current.getTracks()) t.stop()
    }
    liveStreamRef.current = null
    setSnackMsg('Live denoise stopped')
    setSnackSev('info')
    setSnackOpen(true)
  }

  const downloadClean = () => {
    if (!cleanUrl) return
    const a = document.createElement('a')
    a.href = cleanUrl
    a.download = 'clean.wav'
    a.click()
  }

  // Draw waveform on canvas for a given audio URL
  async function drawWaveform(url: string, canvas: HTMLCanvasElement | null, color: string) {
    if (!canvas) return
    try {
      const dpr = window.devicePixelRatio || 1
      const cssWidth = canvas.clientWidth || canvas.parentElement?.clientWidth || 600
      const cssHeight = canvas.clientHeight || 80
      canvas.width = Math.max(300, Math.floor(cssWidth * dpr))
      canvas.height = Math.max(60, Math.floor(cssHeight * dpr))
      const ctx = canvas.getContext('2d')!
      ctx.scale(dpr, dpr)
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Lazy create AudioContext
      if (!audioCtxRef.current) {
        const AC = (window as any).AudioContext || (window as any).webkitAudioContext
        audioCtxRef.current = new AC()
      }
      const ac = audioCtxRef.current!
      const res = await fetch(url)
      const buf = await res.arrayBuffer()
      const audioBuffer = await ac.decodeAudioData(buf)
      const channel = audioBuffer.getChannelData(0)

      // Compute peaks
      const samples = channel.length
      const width = Math.floor((canvas.width / dpr))
      const height = Math.floor((canvas.height / dpr))
      const blockSize = Math.max(1, Math.floor(samples / width))
      const step = blockSize
      const amp = height / 2 - 2

      ctx.lineWidth = 1
      ctx.strokeStyle = color
      ctx.beginPath()
      ctx.moveTo(0, height / 2)
      for (let i = 0; i < width; i++) {
        const start = i * step
        let min = 1.0
        let max = -1.0
        for (let j = 0; j < step && start + j < samples; j++) {
          const sample = channel[start + j]
          if (sample > max) max = sample
          if (sample < min) min = sample
        }
        const yMax = height / 2 - (max * amp)
        const yMin = height / 2 - (min * amp)
        ctx.moveTo(i, yMax)
        ctx.lineTo(i, yMin)
      }
      ctx.stroke()
    } catch (e) {
      // ignore drawing errors
    }
  }

  useEffect(() => {
    if (rawUrl) drawWaveform(rawUrl, rawCanvasRef.current, '#90caf9')
  }, [rawUrl])
  useEffect(() => {
    if (cleanUrl) drawWaveform(cleanUrl, cleanCanvasRef.current, '#80cbc4')
  }, [cleanUrl])

  return (
    <Box>
      <AppBar position="static" color="transparent" sx={{
        backdropFilter: 'blur(10px)',
        borderBottom: '1px solid rgba(255,255,255,0.08)'
      }}>
        <Toolbar>
          <GraphicEqIcon color="primary" sx={{ mr: 1 }} />
          <Typography variant="h6" sx={{ flexGrow: 1 }}>Noise Cancellation Studio</Typography>
          <Button color="primary" href="https://fastapi.tiangolo.com" target="_blank">API Docs</Button>
        </Toolbar>
      </AppBar>

      <Container maxWidth="md" sx={{ py: 4 }}>
        <Card variant="outlined" sx={{ mb: 3 }}>
          <CardContent>
            <Paper
              variant="outlined"
              sx={{
                p: 3,
                borderStyle: 'dashed',
                borderColor: dragOver ? 'primary.main' : 'divider',
                bgcolor: dragOver ? 'action.hover' : 'transparent',
                textAlign: 'center',
                transition: 'all .15s ease-in-out'
              }}
              onDrop={onDropFile}
              onDragOver={onDragOver}
              onDragEnter={onDragOver}
              onDragLeave={onDragLeave}
            >
              <Typography variant="subtitle1" gutterBottom>
                Drag & drop a WAV file here
              </Typography>
              <Typography color="text.secondary" gutterBottom>
                or
              </Typography>
              <Button variant="contained" component="label" startIcon={<UploadIcon />}>
                Choose WAV
                <input hidden type="file" accept="audio/wav" onChange={onPick} />
              </Button>
              <Typography sx={{ mt: 1 }} color="text.secondary">{file?.name ?? 'No file selected'}</Typography>
            </Paper>

            <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} alignItems="center" sx={{ mt: 2 }}>
              <Box sx={{ flexGrow: 1 }} />
             
              <Button disabled={disableFileActions || !file} variant="contained" onClick={doDenoise} startIcon={<CleaningServicesIcon />}>Denoise</Button>
              <Button variant={liveOn ? 'contained' : 'outlined'} color={liveOn ? 'error' : 'primary'} onClick={liveOn ? stopLive : startLive} startIcon={<GraphicEqIcon />}>{liveOn ? 'Stop Live' : 'Live Denoise'}</Button>
              <Button disabled={!cleanUrl} variant="text" onClick={downloadClean} startIcon={<DownloadIcon />}>Download</Button>
            </Stack>

            <Accordion sx={{ mt: 2 }}>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography>Settings</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} alignItems="center">
                  <Typography sx={{ minWidth: 160 }}>Cancellation (%)</Typography>
                  <Slider
                    value={cancelPct}
                    onChange={(_, v) => setCancelPct(Array.isArray(v) ? v[0] : v)}
                    min={0}
                    max={100}
                    step={1}
                    marks={[{ value: 0, label: '0' }, { value: 25, label: '25' }, { value: 50, label: '50' }, { value: 75, label: '75' }, { value: 100, label: '100' }]}
                    sx={{ flex: 1 }}
                  />
                  <Typography sx={{ width: 56, textAlign: 'right' }}>{cancelPct}%</Typography>
                </Stack>
              </AccordionDetails>
            </Accordion>

            {(predicting || denoising) && <LinearProgress sx={{ mt: 2 }} />}
          </CardContent>
        </Card>

        <Stack direction={{ xs: 'column', md: 'row' }} spacing={3}>
          <Card variant="outlined" sx={{ flex: 1 }}>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>Original</Typography>
              <Box sx={{ width: '100%', mb: 1 }}>
                <canvas ref={rawCanvasRef} style={{ width: '100%', height: 80 }} />
              </Box>
              {rawUrl ? (
                <audio ref={audioRawRef} src={rawUrl} controls style={{ width: '100%' }} />
              ) : (
                <Typography color="text.secondary">Upload a WAV file to preview</Typography>
              )}
            </CardContent>
          </Card>
          <Card variant="outlined" sx={{ flex: 1 }}>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>Cleaned</Typography>
              <Box sx={{ width: '100%', mb: 1 }}>
                <canvas ref={cleanCanvasRef} style={{ width: '100%', height: 80 }} />
              </Box>
              {cleanUrl ? (
                <audio ref={audioCleanRef} src={cleanUrl} controls style={{ width: '100%' }} />
              ) : (
                <Typography color="text.secondary">Run Denoise to generate output</Typography>
              )}
            </CardContent>
          </Card>
        </Stack>

  
      </Container>

      <Snackbar open={snackOpen} autoHideDuration={3000} onClose={() => setSnackOpen(false)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}>
        <Alert onClose={() => setSnackOpen(false)} severity={snackSev} sx={{ width: '100%' }}>
          {snackMsg}
        </Alert>
      </Snackbar>
    </Box>
  )
}


