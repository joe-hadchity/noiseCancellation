import sys
import os
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import noisereduce as nr
from tensorflow.keras.models import load_model

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class NoiseCancellerUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Noise Cancellation UI")
        self.setMinimumSize(800, 500)

        # Config (defaults aligned with livetest.py)
        self.sample_rate = 44100
        self.duration_seconds = 5
        self.model_path = "model.h5"
        self.noise_dir = "noise"
        self.output_file = "clean.wav"
        self.prop_decrease = 50  # percentage for UI, converted to 0-1

        # Runtime state
        self.model = None
        self.recorded_audio = None
        self.cleaned_audio = None
        self.pred_probs = None

        self.classes = {
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

        self._build_ui()
        self._load_model_async()

    # ----------------------- UI SETUP -----------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Model / settings group
        settings_group = QGroupBox("Settings")
        settings_layout = QGridLayout()
        settings_group.setLayout(settings_layout)

        settings_layout.addWidget(QLabel("Model path:"), 0, 0)
        self.model_path_edit = QLineEdit(self.model_path)
        browse_model_btn = QPushButton("Browse…")
        browse_model_btn.clicked.connect(self._browse_model)
        settings_layout.addWidget(self.model_path_edit, 0, 1)
        settings_layout.addWidget(browse_model_btn, 0, 2)

        settings_layout.addWidget(QLabel("Duration (s):"), 1, 0)
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 30)
        self.duration_spin.setValue(self.duration_seconds)
        settings_layout.addWidget(self.duration_spin, 1, 1)

        settings_layout.addWidget(QLabel("Sample rate:"), 2, 0)
        self.sr_spin = QSpinBox()
        self.sr_spin.setRange(8000, 96000)
        self.sr_spin.setSingleStep(1000)
        self.sr_spin.setValue(self.sample_rate)
        settings_layout.addWidget(self.sr_spin, 2, 1)

        settings_layout.addWidget(QLabel("Noise reduction (%):"), 3, 0)
        self.prop_slider = QSlider(Qt.Horizontal)
        self.prop_slider.setRange(0, 100)
        self.prop_slider.setValue(self.prop_decrease)
        self.prop_value_label = QLabel(f"{self.prop_decrease}")
        self.prop_slider.valueChanged.connect(
            lambda v: self.prop_value_label.setText(str(v))
        )
        slider_row = QHBoxLayout()
        slider_row.addWidget(self.prop_slider)
        slider_row.addWidget(self.prop_value_label)
        settings_layout.addLayout(slider_row, 3, 1, 1, 2)

        root.addWidget(settings_group)

        # IO group
        io_group = QGroupBox("Audio")
        io_layout = QHBoxLayout()
        io_group.setLayout(io_layout)

        self.record_btn = QPushButton("Record")
        self.record_btn.clicked.connect(self._record_audio)
        self.load_btn = QPushButton("Load WAV…")
        self.load_btn.clicked.connect(self._load_audio)
        self.play_raw_btn = QPushButton("Play Raw")
        self.play_raw_btn.clicked.connect(lambda: self._play_array(self.recorded_audio))
        self.play_clean_btn = QPushButton("Play Clean")
        self.play_clean_btn.clicked.connect(lambda: self._play_array(self.cleaned_audio))

        io_layout.addWidget(self.record_btn)
        io_layout.addWidget(self.load_btn)
        io_layout.addWidget(self.play_raw_btn)
        io_layout.addWidget(self.play_clean_btn)

        root.addWidget(io_group)

        # Actions group
        actions_group = QGroupBox("Process")
        actions_layout = QHBoxLayout()
        actions_group.setLayout(actions_layout)

        self.predict_btn = QPushButton("Predict Noises")
        self.predict_btn.clicked.connect(self._predict_noises)
        self.denoise_btn = QPushButton("Denoise")
        self.denoise_btn.clicked.connect(self._denoise)
        self.save_btn = QPushButton("Save Clean as…")
        self.save_btn.clicked.connect(self._save_clean)

        actions_layout.addWidget(self.predict_btn)
        actions_layout.addWidget(self.denoise_btn)
        actions_layout.addWidget(self.save_btn)

        root.addWidget(actions_group)

        # Status / output
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

    # ----------------------- UTIL -----------------------
    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select model.h5", os.getcwd(), "Model (*.h5 *.keras)")
        if path:
            self.model_path_edit.setText(path)
            self._load_model_async()

    def _set_busy(self, busy: bool):
        for w in [
            self.record_btn, self.load_btn, self.play_raw_btn, self.play_clean_btn,
            self.predict_btn, self.denoise_btn, self.save_btn,
            self.duration_spin, self.sr_spin, self.prop_slider
        ]:
            w.setEnabled(not busy)

    def _notify(self, text: str):
        self.status_label.setText(text)

    def _error(self, text: str):
        self._notify(text)
        QMessageBox.critical(self, "Error", text)

    # ----------------------- MODEL -----------------------
    def _load_model_async(self):
        def _load():
            try:
                self._notify("Loading model…")
                self.model = load_model(self.model_path_edit.text())
                self._notify("Model loaded.")
            except Exception as exc:
                self.model = None
                self._error(f"Failed to load model: {exc}")

        threading.Thread(target=_load, daemon=True).start()

    # ----------------------- AUDIO IO -----------------------
    def _record_audio(self):
        if self.model is None:
            self._error("Model not loaded yet.")
            return

        self.duration_seconds = int(self.duration_spin.value())
        self.sample_rate = int(self.sr_spin.value())

        def _rec():
            try:
                self._set_busy(True)
                self._notify(f"Recording for {self.duration_seconds} seconds…")
                audio = sd.rec(int(self.duration_seconds * self.sample_rate), samplerate=self.sample_rate, channels=1)
                sd.wait()
                self.recorded_audio = audio.flatten().astype(np.float32)
                self._notify("Recording finished.")
            except Exception as exc:
                self._error(f"Recording failed: {exc}")
            finally:
                self._set_busy(False)

        threading.Thread(target=_rec, daemon=True).start()

    def _load_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select WAV", os.getcwd(), "Audio (*.wav)")
        if not path:
            return

        def _load():
            try:
                self._set_busy(True)
                y, sr = librosa.load(path, sr=int(self.sr_spin.value()))
                self.sample_rate = sr
                self.recorded_audio = y.astype(np.float32)
                self._notify(f"Loaded {os.path.basename(path)} at {sr} Hz.")
            except Exception as exc:
                self._error(f"Failed to load audio: {exc}")
            finally:
                self._set_busy(False)

        threading.Thread(target=_load, daemon=True).start()

    def _play_array(self, arr: np.ndarray | None):
        if arr is None:
            self._error("Nothing to play.")
            return

        def _play():
            try:
                sd.play(arr, self.sample_rate)
                sd.wait()
            except Exception as exc:
                self._error(f"Playback failed: {exc}")

        threading.Thread(target=_play, daemon=True).start()

    # ----------------------- PIPELINE -----------------------
    def _extract_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
        melspectrogram = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40, fmax=8000).T, axis=0)
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=40).T, axis=0)
        chroma_cq = np.mean(librosa.feature.chroma_cqt(y=audio, sr=sr, n_chroma=40, bins_per_octave=40).T, axis=0)
        chroma_cens = np.mean(librosa.feature.chroma_cens(y=audio, sr=sr, n_chroma=40, bins_per_octave=40).T, axis=0)
        features = np.vstack((mfccs, melspectrogram, chroma_stft, chroma_cq, chroma_cens)).T
        return features.reshape(1, 40, 5, 1)

    def _collect_reference_noises(self, top_indices: list[int]) -> np.ndarray:
        noise_files: list[str] = []
        for idx in top_indices:
            class_name = self.classes[idx]
            # mirror livetest.py naming but include current files present
            candidates = [
                f"{self.noise_dir}/{class_name.lower().replace(' ', '')}{i}.wav" for i in range(1, 3)
            ]
            # also support existing filenames in noise dir
            if os.path.isdir(self.noise_dir):
                for fname in os.listdir(self.noise_dir):
                    lower = fname.lower()
                    if any(key in lower for key in [
                        'wind', 'horn', 'children', 'bark', 'drill', 'engine', 'gun', 'jack', 'siren', 'street'
                    ]):
                        cand_path = os.path.join(self.noise_dir, fname)
                        if cand_path not in candidates:
                            candidates.append(cand_path)
            for nf in candidates:
                if os.path.exists(nf):
                    noise_files.append(nf)

        all_noise = np.array([], dtype=np.float32)
        for nf in noise_files:
            try:
                noise_clip, _ = librosa.load(nf, sr=self.sample_rate)
                noise_clip = noise_clip.astype(np.float32)
                all_noise = np.concatenate([all_noise, noise_clip]) if all_noise.size else noise_clip
            except Exception:
                continue
        return all_noise

    def _predict_noises(self):
        if self.model is None:
            self._error("Model not loaded.")
            return
        if self.recorded_audio is None:
            self._error("Record or load audio first.")
            return

        def _predict():
            try:
                self._set_busy(True)
                self._notify("Extracting features and predicting…")
                x = self._extract_features(self.recorded_audio, self.sample_rate)
                preds = self.model.predict(x)[0]
                self.pred_probs = preds
                top_indices = preds.argsort()[-3:][::-1]
                text = "Predicted noises:\n" + "\n".join([
                    f"{self.classes[idx]} ({preds[idx]:.3f})" for idx in top_indices
                ])
                self._notify(text)
            except Exception as exc:
                self._error(f"Prediction failed: {exc}")
            finally:
                self._set_busy(False)

        threading.Thread(target=_predict, daemon=True).start()

    def _denoise(self):
        if self.recorded_audio is None:
            self._error("Record or load audio first.")
            return
        if self.model is None:
            self._error("Model not loaded.")
            return

        def _run():
            try:
                self._set_busy(True)
                self._notify("Running denoise…")

                # Predict if not done
                if self.pred_probs is None:
                    x = self._extract_features(self.recorded_audio, self.sample_rate)
                    preds = self.model.predict(x)[0]
                else:
                    preds = self.pred_probs

                top_indices = preds.argsort()[-3:][::-1]
                all_noise = self._collect_reference_noises(list(top_indices))

                prop = float(self.prop_slider.value()) / 100.0
                if all_noise.size > 0:
                    reduced = nr.reduce_noise(
                        y=self.recorded_audio,
                        y_noise=all_noise,
                        sr=self.sample_rate,
                        prop_decrease=prop,
                    )
                    self.cleaned_audio = reduced.astype(np.float32)
                    self._notify("Denoise complete. You can play or save the clean audio.")
                else:
                    self.cleaned_audio = self.recorded_audio.copy()
                    self._notify("No reference noise files found. Using original audio.")
            except Exception as exc:
                self._error(f"Denoise failed: {exc}")
            finally:
                self._set_busy(False)

        threading.Thread(target=_run, daemon=True).start()

    def _save_clean(self):
        if self.cleaned_audio is None:
            self._error("Nothing to save. Run Denoise first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Clean WAV", self.output_file, "WAV (*.wav)")
        if not path:
            return
        try:
            sf.write(path, self.cleaned_audio, self.sample_rate)
            self._notify(f"Saved: {os.path.basename(path)}")
        except Exception as exc:
            self._error(f"Save failed: {exc}")


def main():
    app = QApplication(sys.argv)
    win = NoiseCancellerUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


