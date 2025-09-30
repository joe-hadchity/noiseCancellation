import numpy as np
import librosa
import soundfile as sf
import os
from tensorflow.keras.models import load_model
import noisereduce as nr

# ==========================
# Load trained model
# ==========================
model = load_model('model.h5')
print("\nModel loaded successfully!\n")

# ==========================
# Define classes
# ==========================
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

# ==========================
# Get input file
# ==========================
filename = input("Enter path to audio file: ")
y, sr = librosa.load(filename, sr=None)  # keep original sample rate

# ==========================
# Extract features
# ==========================
mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=8000).T, axis=0)
chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=40).T, axis=0)
chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=40, bins_per_octave=40).T, axis=0)
chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=40, bins_per_octave=40).T, axis=0)

features = np.vstack((mfccs, melspectrogram, chroma_stft, chroma_cq, chroma_cens)).T  # shape (40,5)
x_test = features.reshape(1, 40, 5, 1)

# ==========================
# Make prediction
# ==========================
preds = model.predict(x_test)[0]

# Get top 3 predicted noises
top_indices = preds.argsort()[-3:][::-1]
print("\nPredicted noises:")
for idx in top_indices:
    print(f"{classes[idx]} ({preds[idx]:.3f})")

# ==========================
# Denoising function
# ==========================
def denoise_audio(data, noise_file, output_file="clean.wav"):
    noise, sr_noise = librosa.load(noise_file, sr=sr)  # match main audio sr
    if len(noise.shape) > 1:
        noise = np.mean(noise, axis=1)  # convert to mono if stereo
    reduced = nr.reduce_noise(y=data, y_noise=noise, sr=sr, prop_decrease=0.9)
    sf.write(output_file, reduced, sr)
    print(f"Denoised audio saved as {output_file}")

# ==========================
# Denoise based on prediction
# ==========================
noise_dir = "noise"  # folder containing your reference noise files
for idx in top_indices:
    # Format noise filenames, e.g., engineidling1.wav, engineidling2.wav
    base_name = classes[idx].lower().replace(" ", "")
    noise_files = [f"{noise_dir}/{base_name}{i}.wav" for i in range(1, 3)]
    for nf in noise_files:
        if os.path.exists(nf):
            denoise_audio(y, nf)
