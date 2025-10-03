from __future__ import annotations

import io
import os
from typing import List, Tuple

import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr

from .deps import load_shared_model, get_noise_dir


CLASSES = {
    0: 'Windy',
    1: 'Horn',
    2: 'Children-noise',
    3: 'Dog Bark',
    4: 'Drilling',
    5: 'Engine Idling',
    6: 'Gun Shot',
    7: 'Jackhammer',
    8: 'Siren',
    9: 'Street music'
}


def extract_features(audio: np.ndarray, sr: int) -> np.ndarray:
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    melspectrogram = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40, fmax=8000).T, axis=0)
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=40).T, axis=0)
    chroma_cq = np.mean(librosa.feature.chroma_cqt(y=audio, sr=sr, n_chroma=40, bins_per_octave=40).T, axis=0)
    chroma_cens = np.mean(librosa.feature.chroma_cens(y=audio, sr=sr, n_chroma=40, bins_per_octave=40).T, axis=0)
    features = np.vstack((mfccs, melspectrogram, chroma_stft, chroma_cq, chroma_cens)).T
    return features.reshape(1, 40, 5, 1)


def predict_noises(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, List[int]]:
    model = load_shared_model()
    x = extract_features(audio, sr)
    preds = model.predict(x)[0]
    top_indices = preds.argsort()[-3:][::-1].tolist()
    return preds, top_indices


def list_reference_noises() -> List[str]:
    noise_dir = get_noise_dir()
    results: List[str] = []
    if os.path.isdir(noise_dir):
        for fname in os.listdir(noise_dir):
            lower = fname.lower()
            if lower.endswith('.wav'):
                results.append(os.path.join(noise_dir, fname))
    return results


def collect_reference_noises(top_indices: List[int], sr: int) -> np.ndarray:
    noise_dir = get_noise_dir()
    noise_files: List[str] = []
    for idx in top_indices:
        class_name = CLASSES[idx]
        candidates = [
            os.path.join(noise_dir, f"{class_name.lower().replace(' ', '')}{i}.wav") for i in range(1, 3)
        ]
        if os.path.isdir(noise_dir):
            for fname in os.listdir(noise_dir):
                lower = fname.lower()
                if any(key in lower for key in [
                    'wind', 'horn', 'children', 'bark', 'drill', 'engine', 'gun', 'jack', 'siren', 'street'
                ]):
                    cand_path = os.path.join(noise_dir, fname)
                    if cand_path not in candidates:
                        candidates.append(cand_path)
        for nf in candidates:
            if os.path.exists(nf):
                noise_files.append(nf)

    all_noise = np.array([], dtype=np.float32)
    for nf in noise_files:
        try:
            noise_clip, _ = librosa.load(nf, sr=sr)
            noise_clip = noise_clip.astype(np.float32)
            all_noise = np.concatenate([all_noise, noise_clip]) if all_noise.size else noise_clip
        except Exception:
            continue
    return all_noise


def denoise_with_refs(audio: np.ndarray, sr: int, prop_decrease: float = 0.5) -> np.ndarray:
    _, top_indices = predict_noises(audio, sr)
    all_noise = collect_reference_noises(top_indices, sr)
    if all_noise.size == 0:
        return audio.copy()
    reduced = nr.reduce_noise(y=audio, y_noise=all_noise, sr=sr, prop_decrease=float(prop_decrease))
    return reduced.astype(np.float32)


def load_audio_from_bytes(data: bytes, target_sr: int | None = None) -> Tuple[np.ndarray, int]:
    buf = io.BytesIO(data)
    y, sr = librosa.load(buf, sr=target_sr)
    return y.astype(np.float32), int(sr)


def write_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    y = y.astype(np.float32)
    out = io.BytesIO()
    sf.write(out, y, sr, format='WAV', subtype='PCM_16')
    return out.getvalue()





