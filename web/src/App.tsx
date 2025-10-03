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

  const enableActions = !!file && !predicting && !denoising

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
      const res = await axios.post(`${API_BASE}/denoise`, fd, {
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
              <Button disabled={!enableActions} variant="outlined" onClick={doPredict} startIcon={<GraphicEqIcon />}>Predict</Button>
              <Button disabled={!enableActions} variant="contained" onClick={doDenoise} startIcon={<CleaningServicesIcon />}>Denoise</Button>
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

        <Card variant="outlined" sx={{ mt: 3 }}>
          <CardContent>
            <Typography variant="subtitle1" gutterBottom>Predicted Noises</Typography>
            {labels && probs && topIdx ? (
              <Stack spacing={1}>
                {labels.map((label, i) => {
                  const idx = topIdx[i]
                  const p = typeof idx === 'number' ? probs[idx] : 0
                  const val = Math.max(0, Math.min(100, (p ?? 0) * 100))
                  if (val <= 0) return null
                  return (
                    <Stack key={`${label}-${i}`} direction="row" spacing={2} alignItems="center">
                      <Typography sx={{ width: 160 }}>{label}</Typography>
                      <LinearProgress variant="determinate" value={val} sx={{ flex: 1 }} />
                      <Typography sx={{ width: 60, textAlign: 'right' }}>{val.toFixed(1)}%</Typography>
                    </Stack>
                  )
                })}
              </Stack>
            ) : (
              <Typography color="text.secondary">Run Predict to see results.</Typography>
            )}
          </CardContent>
        </Card>
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


