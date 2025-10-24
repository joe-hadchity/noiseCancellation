"""
Virtual Microphone with Noise Cancellation
Creates a virtual microphone that applies noise cancellation to audio input
"""

import numpy as np
import sounddevice as sd
import threading
import queue
import time
import librosa
import noisereduce as nr
from typing import Optional, Callable
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VirtualMicrophone:
    def __init__(self, 
                 input_device: Optional[int] = None,
                 output_device: Optional[int] = None,
                 sample_rate: int = 44100,
                 chunk_size: int = 1024,
                 noise_reduction_strength: float = 0.5,
                 model_path: str = "model.h5"):
        """
        Initialize the virtual microphone
        
        Args:
            input_device: Input device index (None for default)
            output_device: Output device index (None for default) 
            sample_rate: Audio sample rate
            chunk_size: Size of audio chunks to process
            noise_reduction_strength: Strength of noise reduction (0.0 to 1.0)
            model_path: Path to the noise classification model
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.noise_reduction_strength = noise_reduction_strength
        self.model_path = model_path
        
        # Audio queues for real-time processing
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)
        
        # Threading control
        self.is_running = False
        self.input_thread = None
        self.processing_thread = None
        self.output_thread = None
        
        # Audio devices
        self.input_device = input_device
        self.output_device = output_device
        
        # Load noise classification model
        self.model = None
        self.load_model()
        
        # Noise classes for reference
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
        
        # Reference noise directory
        self.noise_dir = Path("noise")
        
    def load_model(self):
        """Load the noise classification model"""
        try:
            from tensorflow.keras.models import load_model
            if Path(self.model_path).exists():
                self.model = load_model(self.model_path)
                logger.info(f"Loaded model from {self.model_path}")
            else:
                logger.warning(f"Model file {self.model_path} not found. Using basic noise reduction only.")
        except Exception as e:
            logger.warning(f"Could not load model: {e}. Using basic noise reduction only.")
    
    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract features for noise classification"""
        try:
            mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=40).T, axis=0)
            melspectrogram = np.mean(librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, n_mels=40, fmax=8000).T, axis=0)
            chroma_stft = np.mean(librosa.feature.chroma_stft(y=audio, sr=self.sample_rate, n_chroma=40).T, axis=0)
            chroma_cq = np.mean(librosa.feature.chroma_cqt(y=audio, sr=self.sample_rate, n_chroma=40, bins_per_octave=40).T, axis=0)
            chroma_cens = np.mean(librosa.feature.chroma_cens(y=audio, sr=self.sample_rate, n_chroma=40, bins_per_octave=40).T, axis=0)
            
            features = np.vstack((mfccs, melspectrogram, chroma_stft, chroma_cq, chroma_cens)).T
            return features.reshape(1, 40, 5, 1)
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    def predict_noises(self, audio: np.ndarray) -> tuple:
        """Predict noise types in audio"""
        if self.model is None:
            return None, []
        
        try:
            features = self.extract_features(audio)
            if features is None:
                return None, []
            
            preds = self.model.predict(features, verbose=0)[0]
            top_indices = preds.argsort()[-3:][::-1].tolist()
            return preds, top_indices
        except Exception as e:
            logger.error(f"Noise prediction failed: {e}")
            return None, []
    
    def collect_reference_noises(self, top_indices: list) -> np.ndarray:
        """Collect reference noise samples for denoising"""
        noise_files = []
        
        for idx in top_indices:
            class_name = self.classes[idx]
            # Try different naming patterns
            candidates = [
                f"{class_name.lower().replace(' ', '')}{i}.wav" for i in range(1, 3)
            ]
            
            # Also search for files with keywords
            if self.noise_dir.exists():
                for fname in self.noise_dir.iterdir():
                    if fname.suffix.lower() == '.wav':
                        lower_name = fname.name.lower()
                        keywords = ['wind', 'horn', 'children', 'bark', 'drill', 'engine', 'gun', 'jack', 'siren', 'street']
                        if any(keyword in lower_name for keyword in keywords):
                            candidates.append(fname.name)
            
            # Add existing files
            for candidate in candidates:
                candidate_path = self.noise_dir / candidate
                if candidate_path.exists() and candidate_path not in noise_files:
                    noise_files.append(str(candidate_path))
        
        # Load and concatenate noise samples
        all_noise = np.array([], dtype=np.float32)
        for noise_file in noise_files:
            try:
                noise_clip, _ = librosa.load(noise_file, sr=self.sample_rate)
                noise_clip = noise_clip.astype(np.float32)
                all_noise = np.concatenate([all_noise, noise_clip]) if all_noise.size else noise_clip
            except Exception as e:
                logger.warning(f"Could not load noise file {noise_file}: {e}")
                continue
        
        return all_noise
    
    def denoise_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise reduction to audio"""
        try:
            # Try model-based denoising first
            if self.model is not None:
                preds, top_indices = self.predict_noises(audio)
                if preds is not None and top_indices:
                    all_noise = self.collect_reference_noises(top_indices)
                    if all_noise.size > 0:
                        reduced = nr.reduce_noise(
                            y=audio, 
                            y_noise=all_noise, 
                            sr=self.sample_rate, 
                            prop_decrease=self.noise_reduction_strength
                        )
                        return reduced.astype(np.float32)
            
            # Fallback to basic noise reduction
            reduced = nr.reduce_noise(
                y=audio, 
                sr=self.sample_rate, 
                prop_decrease=self.noise_reduction_strength,
                stationary=True
            )
            return reduced.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Denoising failed: {e}")
            return audio.astype(np.float32)
    
    def input_callback(self, indata, frames, time, status):
        """Callback for audio input"""
        if status:
            logger.warning(f"Input status: {status}")
        
        try:
            # Convert to mono and add to queue
            audio_data = indata[:, 0] if indata.ndim > 1 else indata
            self.input_queue.put_nowait(audio_data.copy())
        except queue.Full:
            logger.warning("Input queue full, dropping audio chunk")
    
    def processing_worker(self):
        """Worker thread for audio processing"""
        logger.info("Audio processing thread started")
        
        while self.is_running:
            try:
                # Get audio chunk from input queue
                audio_chunk = self.input_queue.get(timeout=0.1)
                
                # Apply noise reduction
                cleaned_audio = self.denoise_audio(audio_chunk)
                
                # Put processed audio in output queue
                try:
                    self.output_queue.put_nowait(cleaned_audio)
                except queue.Full:
                    logger.warning("Output queue full, dropping processed chunk")
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing error: {e}")
        
        logger.info("Audio processing thread stopped")
    
    def output_callback(self, outdata, frames, time, status):
        """Callback for audio output"""
        if status:
            logger.warning(f"Output status: {status}")
        
        try:
            # Get processed audio from queue
            audio_chunk = self.output_queue.get_nowait()
            
            # Ensure correct shape for output
            if audio_chunk.ndim == 1:
                outdata[:, 0] = audio_chunk[:frames]
                if outdata.shape[1] > 1:  # Stereo output
                    outdata[:, 1] = audio_chunk[:frames]
            else:
                outdata[:] = audio_chunk[:frames]
                
        except queue.Empty:
            # No processed audio available, output silence
            outdata.fill(0)
        except Exception as e:
            logger.error(f"Output callback error: {e}")
            outdata.fill(0)
    
    def start(self):
        """Start the virtual microphone"""
        if self.is_running:
            logger.warning("Virtual microphone is already running")
            return
        
        logger.info("Starting virtual microphone...")
        self.is_running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.processing_worker, daemon=True)
        self.processing_thread.start()
        
        # Start audio streams
        try:
            self.input_stream = sd.InputStream(
                device=self.input_device,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=self.input_callback,
                dtype=np.float32
            )
            
            self.output_stream = sd.OutputStream(
                device=self.output_device,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=self.output_callback,
                dtype=np.float32
            )
            
            self.input_stream.start()
            self.output_stream.start()
            
            logger.info("Virtual microphone started successfully")
            logger.info(f"Input device: {self.input_stream.device}")
            logger.info(f"Output device: {self.output_stream.device}")
            
        except Exception as e:
            logger.error(f"Failed to start audio streams: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop the virtual microphone"""
        if not self.is_running:
            return
        
        logger.info("Stopping virtual microphone...")
        self.is_running = False
        
        # Stop audio streams
        try:
            if hasattr(self, 'input_stream'):
                self.input_stream.stop()
                self.input_stream.close()
            if hasattr(self, 'output_stream'):
                self.output_stream.stop()
                self.output_stream.close()
        except Exception as e:
            logger.error(f"Error stopping audio streams: {e}")
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        logger.info("Virtual microphone stopped")
    
    def list_audio_devices(self):
        """List available audio devices"""
        devices = sd.query_devices()
        logger.info("Available audio devices:")
        for i, device in enumerate(devices):
            logger.info(f"  {i}: {device['name']} ({device['hostapi']})")
            logger.info(f"     Input channels: {device['max_input_channels']}")
            logger.info(f"     Output channels: {device['max_output_channels']}")
            logger.info(f"     Sample rate: {device['default_samplerate']}")


def main():
    """Main function to run the virtual microphone"""
    # Create virtual microphone instance
    vm = VirtualMicrophone(
        sample_rate=44100,
        chunk_size=1024,
        noise_reduction_strength=0.5
    )
    
    # List available devices
    vm.list_audio_devices()
    
    try:
        # Start the virtual microphone
        vm.start()
        
        # Keep running until interrupted
        logger.info("Virtual microphone is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        vm.stop()


if __name__ == "__main__":
    main()
