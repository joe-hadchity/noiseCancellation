import sounddevice as sd
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import noisereduce as nr
import os
import soundfile as sf

# Config
DURATION = 5
SR = 44100
NOISE_DIR = "noise"
MODEL_PATH = "model.h5"
OUTPUT_FILE = "clean.wav"
PROP_DECREASE = 0.5

# Load model
model = load_model(MODEL_PATH)

classes = {
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

# Record audio
print(f"Recording for {DURATION} seconds...")
audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1)
sd.wait()
audio = audio.flatten()
print("Recording finished!")

# Feature extraction
mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=40).T, axis=0)
melspectrogram = np.mean(librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=40, fmax=8000).T, axis=0)
chroma_stft = np.mean(librosa.feature.chroma_stft(y=audio, sr=SR, n_chroma=40).T, axis=0)
chroma_cq = np.mean(librosa.feature.chroma_cqt(y=audio, sr=SR, n_chroma=40, bins_per_octave=40).T, axis=0)
chroma_cens = np.mean(librosa.feature.chroma_cens(y=audio, sr=SR, n_chroma=40, bins_per_octave=40).T, axis=0)

features = np.vstack((mfccs, melspectrogram, chroma_stft, chroma_cq, chroma_cens)).T
x_test = features.reshape(1, 40, 5, 1)

# Predict noises
preds = model.predict(x_test)[0]
top_indices = preds.argsort()[-3:][::-1]
print("\nPredicted noises:")
for idx in top_indices:
    print(f"{classes[idx]} ({preds[idx]:.3f})")

# Collect all noise reference files
noise_files = []
for idx in top_indices:
    for i in range(1, 3):
        nf = f"{NOISE_DIR}/{classes[idx].lower().replace(' ', '')}{i}.wav"
        if os.path.exists(nf):
            noise_files.append(nf)

# Load and concatenate all reference noises
all_noise = np.array([])
for nf in noise_files:
    noise_clip, _ = librosa.load(nf, sr=SR)
    all_noise = np.concatenate([all_noise, noise_clip]) if all_noise.size else noise_clip

# Apply noise reduction once
if all_noise.size > 0:
    reduced = nr.reduce_noise(y=audio, y_noise=all_noise, sr=SR, prop_decrease=PROP_DECREASE)
    sf.write(OUTPUT_FILE, reduced, SR)
    print(f"Denoised audio saved as {OUTPUT_FILE}")
else:
    print("No reference noise files found. Saving original recording.")
    sf.write(OUTPUT_FILE, audio, SR)
