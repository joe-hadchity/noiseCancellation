from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from .audio import (
    load_audio_from_bytes,
    predict_noises,
    denoise_with_refs,
    list_reference_noises,
    write_wav_bytes,
    CLASSES,
)
from .deps import load_shared_model


app = FastAPI(title="Noise Cancellation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/warmup")
async def warmup():
    # Ensure model is loaded
    _ = load_shared_model()
    return {"status": "ready"}


@app.get("/noises")
async def noises():
    files = list_reference_noises()
    return {"count": len(files), "files": files}


@app.post("/predict")
async def predict(audio: UploadFile = File(...), sample_rate: int | None = Form(None)):
    data = await audio.read()
    y, sr = load_audio_from_bytes(data, sample_rate)
    preds, top = predict_noises(y, sr)
    result = {
        "sample_rate": sr,
        "top_indices": top,
        "top_labels": [CLASSES[i] for i in top],
        "probabilities": [float(p) for p in preds.tolist()],
    }
    return JSONResponse(result)


@app.post("/denoise")
async def denoise(audio: UploadFile = File(...), sample_rate: int | None = Form(None), prop_decrease: float = Form(0.5)):
    data = await audio.read()
    y, sr = load_audio_from_bytes(data, sample_rate)
    cleaned = denoise_with_refs(y, sr, prop_decrease=prop_decrease)
    wav_bytes = write_wav_bytes(cleaned, sr)
    return Response(content=wav_bytes, media_type="audio/wav", headers={
        "Content-Disposition": "attachment; filename=clean.wav"
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)





