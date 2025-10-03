import sys
import os
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import noisereduce as nr
from tensorflow.keras.models import load_model

from PyQt5.QtCore import Qt, pyqtSignal
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
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class NoiseCancellerUI(QMainWindow):
    notify_sig = pyqtSignal(str)
    busy_sig = pyqtSignal(bool)
    error_sig = pyqtSignal(str)
    preds_sig = pyqtSignal(object)
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
        self.notify_sig.connect(self._notify)
        self.busy_sig.connect(self._set_busy)
        self.error_sig.connect(self._error)
        self.preds_sig.connect(self._render_predictions)
        self._load_model_async()

    # ----------------------- UI SETUP -----------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Theming (simple dark theme)
        self.setStyleSheet(
            """
            QWidget { background-color: #121212; color: #e0e0e0; }
            QGroupBox { border: 1px solid #333; border-radius: 6px; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; color: #90caf9; }
            QPushButton { background-color: #1f1f1f; border: 1px solid #333; padding: 6px 10px; border-radius: 4px; }
            QPushButton:hover { background-color: #2a2a2a; }
            QPushButton:disabled { color: #777; border-color: #2a2a2a; }
            QLineEdit, QSpinBox, QSlider, QTextEdit { background-color: #1a1a1a; border: 1px solid #333; }
            QSlider::groove:horizontal { height: 6px; background: #333; border-radius: 3px; }
            QSlider::handle:horizontal { background: #90caf9; border: 1px solid #64b5f6; width: 14px; margin: -5px 0; border-radius: 7px; }
            QProgressBar { border: 1px solid #333; border-radius: 4px; text-align: center; }
            QProgressBar::chunk { background-color: #90caf9; }
            """
        )

        # Header
        header = QLabel("Noise Cancellation Studio")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 20px; font-weight: 600; margin: 6px 0 2px 0; color: #bbdefb;")
        subheader = QLabel("Load or record audio, detect noises, and denoise.")
        subheader.setAlignment(Qt.AlignCenter)
        subheader.setStyleSheet("color: #9e9e9e; margin-bottom: 8px;")
        root.addWidget(header)
        root.addWidget(subheader)

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
        self.duration_spin.setToolTip("Recording duration in seconds")
        settings_layout.addWidget(self.duration_spin, 1, 1)

        settings_layout.addWidget(QLabel("Sample rate:"), 2, 0)
        self.sr_spin = QSpinBox()
        self.sr_spin.setRange(8000, 96000)
        self.sr_spin.setSingleStep(1000)
        self.sr_spin.setValue(self.sample_rate)
        self.sr_spin.setToolTip("Target sample rate when recording or loading")
        settings_layout.addWidget(self.sr_spin, 2, 1)

        settings_layout.addWidget(QLabel("Cancellation (%):"), 3, 0)
        self.prop_slider = QSlider(Qt.Horizontal)
        self.prop_slider.setRange(0, 100)
        self.prop_slider.setValue(self.prop_decrease)
        self.prop_value_label = QLabel(f"{self.prop_decrease}%")
        self.prop_slider.valueChanged.connect(
            lambda v: self.prop_value_label.setText(f"{v}%")
        )
        self.prop_slider.setTickPosition(QSlider.TicksBelow)
        self.prop_slider.setTickInterval(10)
        slider_row = QHBoxLayout()
        slider_row.addWidget(self.prop_slider)
        slider_row.addWidget(self.prop_value_label)
        self.prop_slider.setToolTip("How aggressively to cancel detected noises")
        settings_layout.addLayout(slider_row, 3, 1, 1, 2)

        root.addWidget(settings_group)

        # IO group
        io_group = QGroupBox("Audio")
        io_layout = QHBoxLayout()
        io_group.setLayout(io_layout)

        self.record_btn = QPushButton("Record")
        self.record_btn.setToolTip("Record from default microphone")
        self.record_btn.clicked.connect(self._record_audio)
        self.load_btn = QPushButton("Load WAV…")
        self.load_btn.setToolTip("Load an existing .wav file")
        self.load_btn.clicked.connect(self._load_audio)
        self.play_raw_btn = QPushButton("Play Raw")
        self.play_raw_btn.setToolTip("Play the original recorded/loaded audio")
        self.play_raw_btn.clicked.connect(lambda: self._play_array(self.recorded_audio))
        self.play_clean_btn = QPushButton("Play Clean")
        self.play_clean_btn.setToolTip("Play the denoised audio (after processing)")
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
        self.predict_btn.setToolTip("Classify likely noise types in the audio")
        self.predict_btn.clicked.connect(self._predict_noises)
        self.denoise_btn = QPushButton("Denoise")
        self.denoise_btn.setToolTip("Reduce noise using reference samples")
        self.denoise_btn.clicked.connect(self._denoise)
        self.save_btn = QPushButton("Save Clean as…")
        self.save_btn.setToolTip("Export denoised audio as a .wav file")
        self.save_btn.clicked.connect(self._save_clean)

        actions_layout.addWidget(self.predict_btn)
        actions_layout.addWidget(self.denoise_btn)
        actions_layout.addWidget(self.save_btn)

        root.addWidget(actions_group)

        # Status / progress
        status_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate
        self.progress_bar.setVisible(False)
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        status_row.addWidget(self.progress_bar)
        status_row.addWidget(self.status_label, 1)
        root.addLayout(status_row)

        # Predictions panel
        preds_group = QGroupBox("Predictions")
        preds_v = QVBoxLayout()
        preds_group.setLayout(preds_v)
        self.predictions_container = QVBoxLayout()
        self.predictions_container.setSpacing(6)
        self.predictions_container.setContentsMargins(0, 0, 0, 0)
        # placeholder
        self._pred_placeholder = QLabel("Run Predict to see results.")
        self._pred_placeholder.setStyleSheet("color: #9e9e9e")
        self.predictions_container.addWidget(self._pred_placeholder)
        preds_v.addLayout(self.predictions_container)
        root.addWidget(preds_group)

        # Log output
        log_group = QGroupBox("Activity Log")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMinimumHeight(140)
        self.log_edit.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")
        log_layout.addWidget(self.log_edit)
        root.addWidget(log_group)

    # ----------------------- UTIL -----------------------
    def _clear_layout(self, layout: QVBoxLayout):
        try:
            while layout.count():
                item = layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.setParent(None)
        except Exception:
            pass

    def _render_predictions(self, items: object):
        # items expected as list of (label: str, percent: float)
        self._clear_layout(self.predictions_container)
        data = items if isinstance(items, list) else []
        if not data:
            self._pred_placeholder = QLabel("Run Predict to see results.")
            self._pred_placeholder.setStyleSheet("color: #9e9e9e")
            self.predictions_container.addWidget(self._pred_placeholder)
            return
        for label, percent in data:
            row = QHBoxLayout()
            name_lbl = QLabel(str(label))
            bar = QProgressBar()
            bar.setRange(0, 100)
            try:
                bar.setValue(int(round(float(percent))))
            except Exception:
                bar.setValue(0)
            val_lbl = QLabel(f"{float(percent):.1f}%")
            row.addWidget(name_lbl)
            row.addWidget(bar, 1)
            row.addWidget(val_lbl)
            wrap = QWidget()
            wrap.setLayout(row)
            self.predictions_container.addWidget(wrap)

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
        self.progress_bar.setVisible(busy)

    def _notify(self, text: str):
        self.status_label.setText(text)
        try:
            from datetime import datetime
            stamp = datetime.now().strftime("%H:%M:%S")
            self.log_edit.append(f"[{stamp}] {text}")
        except Exception:
            pass

    def _error(self, text: str):
        self._notify(text)
        QMessageBox.critical(self, "Error", text)

    # ----------------------- MODEL -----------------------
    def _load_model_async(self):
        def _load():
            try:
                self.notify_sig.emit("Loading model…")
                self.model = load_model(self.model_path_edit.text())
                self.notify_sig.emit("Model loaded.")
            except Exception as exc:
                self.model = None
                self.error_sig.emit(f"Failed to load model: {exc}")

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
                self.busy_sig.emit(True)
                self.notify_sig.emit(f"Recording for {self.duration_seconds} seconds…")
                audio = sd.rec(int(self.duration_seconds * self.sample_rate), samplerate=self.sample_rate, channels=1)
                sd.wait()
                self.recorded_audio = audio.flatten().astype(np.float32)
                self.notify_sig.emit("Recording finished.")
            except Exception as exc:
                self.error_sig.emit(f"Recording failed: {exc}")
            finally:
                self.busy_sig.emit(False)

        threading.Thread(target=_rec, daemon=True).start()

    def _load_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select WAV", os.getcwd(), "Audio (*.wav)")
        if not path:
            return

        def _load():
            try:
                self.busy_sig.emit(True)
                y, sr = librosa.load(path, sr=int(self.sr_spin.value()))
                self.sample_rate = sr
                self.recorded_audio = y.astype(np.float32)
                self.notify_sig.emit(f"Loaded {os.path.basename(path)} at {sr} Hz.")
            except Exception as exc:
                self.error_sig.emit(f"Failed to load audio: {exc}")
            finally:
                self.busy_sig.emit(False)

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
                self.error_sig.emit(f"Playback failed: {exc}")

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

    def _extract_features_batch(self, audio: np.ndarray, sr: int, window_sec: float = 1.0, hop_sec: float = 0.5) -> np.ndarray:
        # Normalize to avoid scale issues
        try:
            audio = librosa.util.normalize(audio)
        except Exception:
            pass

        win = max(int(window_sec * sr), 1)
        hop = max(int(hop_sec * sr), 1)
        n = audio.shape[0]
        starts = list(range(0, max(n - win + 1, 1), hop))
        if not starts:
            starts = [0]

        feats: list[np.ndarray] = []
        for s in starts:
            seg = audio[s:s + win]
            if seg.shape[0] < win:
                pad = np.zeros(win - seg.shape[0], dtype=seg.dtype)
                seg = np.concatenate([seg, pad])
            f = self._extract_features(seg, sr)
            feats.append(f[0])

        batch = np.stack(feats, axis=0)
        return batch

    def _predict_aggregated(self, audio: np.ndarray, sr: int) -> np.ndarray:
        # Predict on overlapping windows and average probabilities for stability
        if self.model is None:
            raise RuntimeError("Model not loaded")
        batch = self._extract_features_batch(audio, sr)
        preds = self.model.predict(batch)
        if preds.ndim == 2:
            agg = preds.mean(axis=0)
        else:
            agg = preds[0]
        return agg

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
                self.busy_sig.emit(True)
                self.notify_sig.emit("Extracting features and predicting…")
                preds = self._predict_aggregated(self.recorded_audio, self.sample_rate)
                self.pred_probs = preds
                # Build display: show only labels with non-zero percentage (rounded).
                perc = preds * 100.0
                items = [
                    (self.classes[idx], float(perc[idx])) for idx in range(len(perc)) if round(float(perc[idx]), 1) > 0.0
                ]
                items.sort(key=lambda t: t[1], reverse=True)
                self.preds_sig.emit(items)
                # also write a brief summary to the log
                if items:
                    summary = ", ".join([f"{label} {val:.1f}%" for label, val in items[:3]])
                    self.notify_sig.emit(f"Predicted: {summary}")
                else:
                    self.notify_sig.emit("Predicted: (none > 0%)")
            except Exception as exc:
                self.error_sig.emit(f"Prediction failed: {exc}")
            finally:
                self.busy_sig.emit(False)

        threading.Thread(target=_predict, daemon=True).start()

    def _denoise(self):
        if self.recorded_audio is None:
            self._error("Record or load audio first.")
            return
        if self.model is None:
            self._error("Model not Loaded.")
            return

        def _run():
            try:
                self.busy_sig.emit(True)
                self.notify_sig.emit("Running denoise…")

                # Predict if not done (use aggregated prediction)
                if self.pred_probs is None:
                    preds = self._predict_aggregated(self.recorded_audio, self.sample_rate)
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
                    self.notify_sig.emit("Denoise complete. You can play or save the clean audio.")
                else:
                    self.cleaned_audio = self.recorded_audio.copy()
                    self.notify_sig.emit("No reference noise files found. Using original audio.")
            except Exception as exc:
                self.error_sig.emit(f"Denoise failed: {exc}")
            finally:
                self.busy_sig.emit(False)

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
