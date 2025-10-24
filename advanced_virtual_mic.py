"""
Advanced Virtual Microphone with VB-Cable Integration
This creates a virtual microphone that outputs cleaned audio through VB-Cable
"""

import numpy as np
import sounddevice as sd
import threading
import queue
import time
import librosa
import noisereduce as nr
from typing import Optional, List, Dict
import logging
from pathlib import Path
import json
import sys

# Import your existing audio processing modules
try:
    from backend.audio import (
        load_audio_from_bytes,
        predict_noises,
        denoise_with_refs,
        denoise_basic,
        CLASSES
    )
    from backend.deps import load_shared_model
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Backend modules not available, using standalone implementation")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedVirtualMicrophone:
    def __init__(self, 
                 input_device: Optional[int] = None,
                 vbcable_device: Optional[int] = None,
                 sample_rate: int = 44100,
                 chunk_size: int = 1024,
                 noise_reduction_strength: float = 0.5,
                 model_path: str = "model.h5",
                 config_file: str = "vm_config.json"):
        """
        Initialize the advanced virtual microphone
        
        Args:
            input_device: Input device index (None for default microphone)
            vbcable_device: VB-Cable output device index
            sample_rate: Audio sample rate
            chunk_size: Size of audio chunks to process
            noise_reduction_strength: Strength of noise reduction (0.0 to 1.0)
            model_path: Path to the noise classification model
            config_file: Path to configuration file
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.noise_reduction_strength = noise_reduction_strength
        self.model_path = model_path
        self.config_file = config_file
        
        # Load configuration
        self.config = self.load_config()
        
        # Audio queues for real-time processing
        self.input_queue = queue.Queue(maxsize=20)
        self.output_queue = queue.Queue(maxsize=20)
        
        # Threading control
        self.is_running = False
        self.input_thread = None
        self.processing_thread = None
        self.output_thread = None
        
        # Audio devices
        self.input_device = input_device or self.config.get('input_device')
        self.vbcable_device = vbcable_device or self.find_vbcable_device()
        
        # Load model and backend
        self.model = None
        self.backend_available = BACKEND_AVAILABLE
        
        if self.backend_available:
            try:
                self.model = load_shared_model()
                logger.info("Loaded backend model successfully")
            except Exception as e:
                logger.warning(f"Could not load backend model: {e}")
                self.backend_available = False
        
        if not self.backend_available:
            self.load_standalone_model()
        
        # Performance monitoring
        self.stats = {
            'chunks_processed': 0,
            'processing_time': 0,
            'queue_overflows': 0,
            'errors': 0
        }
        
        # Noise classes
        self.classes = CLASSES if BACKEND_AVAILABLE else {
            0: 'Windy', 1: 'Horn', 2: 'Children-noise', 3: 'Dog Bark',
            4: 'Drilling', 5: 'Engine Idling', 6: 'Gun Shot', 7: 'Jackhammer',
            8: 'Siren', 9: 'Street music'
        }
        
        # Reference noise directory
        self.noise_dir = Path("noise")
        
    def load_config(self) -> Dict:
        """Load configuration from file"""
        default_config = {
            'input_device': None,
            'vbcable_device': None,
            'sample_rate': 44100,
            'chunk_size': 1024,
            'noise_reduction_strength': 0.5,
            'enable_aggressive_mode': False,
            'enable_echo_cancellation': True,
            'enable_automatic_gain_control': True
        }
        
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_file}")
                return {**default_config, **config}
            else:
                logger.info("No config file found, using defaults")
                return default_config
        except Exception as e:
            logger.warning(f"Could not load config: {e}, using defaults")
            return default_config
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Could not save config: {e}")
    
    def find_vbcable_device(self) -> Optional[int]:
        """Find VB-Cable device index"""
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if 'cable' in device['name'].lower() and device['max_output_channels'] > 0:
                    logger.info(f"Found VB-Cable device: {device['name']} (index: {i})")
                    return i
            
            logger.warning("VB-Cable device not found")
            return None
        except Exception as e:
            logger.error(f"Error finding VB-Cable device: {e}")
            return None
    
    def load_standalone_model(self):
        """Load model using standalone implementation"""
        try:
            from tensorflow.keras.models import load_model
            if Path(self.model_path).exists():
                self.model = load_model(self.model_path)
                logger.info(f"Loaded standalone model from {self.model_path}")
            else:
                logger.warning(f"Model file {self.model_path} not found")
        except Exception as e:
            logger.warning(f"Could not load standalone model: {e}")
    
    def extract_features(self, audio: np.ndarray) -> Optional[np.ndarray]:
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
        if self.backend_available:
            try:
                return predict_noises(audio, self.sample_rate)
            except Exception as e:
                logger.warning(f"Backend prediction failed: {e}")
        
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
    
    def collect_reference_noises(self, top_indices: List[int]) -> np.ndarray:
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
                if candidate_path.exists() and str(candidate_path) not in noise_files:
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
        start_time = time.time()
        
        try:
            # Try backend denoising first
            if self.backend_available:
                try:
                    cleaned = denoise_with_refs(audio, self.sample_rate, prop_decrease=self.noise_reduction_strength)
                    if cleaned.size > 0:
                        self.stats['chunks_processed'] += 1
                        self.stats['processing_time'] += time.time() - start_time
                        return cleaned.astype(np.float32)
                except Exception as e:
                    logger.warning(f"Backend denoising failed: {e}")
            
            # Try model-based denoising
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
                        self.stats['chunks_processed'] += 1
                        self.stats['processing_time'] += time.time() - start_time
                        return reduced.astype(np.float32)
            
            # Fallback to basic noise reduction
            if self.backend_available:
                try:
                    cleaned = denoise_basic(audio, self.sample_rate, prop_decrease=self.noise_reduction_strength)
                    self.stats['chunks_processed'] += 1
                    self.stats['processing_time'] += time.time() - start_time
                    return cleaned.astype(np.float32)
                except Exception as e:
                    logger.warning(f"Backend basic denoising failed: {e}")
            
            # Standalone basic noise reduction
            reduced = nr.reduce_noise(
                y=audio, 
                sr=self.sample_rate, 
                prop_decrease=self.noise_reduction_strength,
                stationary=True
            )
            self.stats['chunks_processed'] += 1
            self.stats['processing_time'] += time.time() - start_time
            return reduced.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Denoising failed: {e}")
            self.stats['errors'] += 1
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
            self.stats['queue_overflows'] += 1
    
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
                    self.stats['queue_overflows'] += 1
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing error: {e}")
                self.stats['errors'] += 1
        
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
        
        logger.info("Starting advanced virtual microphone...")
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
            
            if self.vbcable_device is not None:
                self.output_stream = sd.OutputStream(
                    device=self.vbcable_device,
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
                logger.info(f"VB-Cable output device: {self.output_stream.device}")
            else:
                logger.error("VB-Cable device not found. Please install VB-Cable first.")
                self.stop()
                raise RuntimeError("VB-Cable device not available")
            
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
        
        # Print statistics
        self.print_stats()
        logger.info("Virtual microphone stopped")
    
    def print_stats(self):
        """Print performance statistics"""
        if self.stats['chunks_processed'] > 0:
            avg_time = self.stats['processing_time'] / self.stats['chunks_processed']
            logger.info(f"Performance Stats:")
            logger.info(f"  Chunks processed: {self.stats['chunks_processed']}")
            logger.info(f"  Average processing time: {avg_time:.4f}s")
            logger.info(f"  Queue overflows: {self.stats['queue_overflows']}")
            logger.info(f"  Errors: {self.stats['errors']}")
    
    def list_audio_devices(self):
        """List available audio devices"""
        devices = sd.query_devices()
        logger.info("Available audio devices:")
        for i, device in enumerate(devices):
            logger.info(f"  {i}: {device['name']} ({device['hostapi']})")
            logger.info(f"     Input channels: {device['max_input_channels']}")
            logger.info(f"     Output channels: {device['max_output_channels']}")
            logger.info(f"     Sample rate: {device['default_samplerate']}")
            
            # Highlight VB-Cable devices
            if 'cable' in device['name'].lower():
                logger.info(f"     *** VB-CABLE DEVICE ***")


def main():
    """Main function to run the advanced virtual microphone"""
    # Create virtual microphone instance
    vm = AdvancedVirtualMicrophone(
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
        logger.info("Advanced virtual microphone is running. Press Ctrl+C to stop.")
        logger.info("Your cleaned audio will be available through VB-Cable!")
        
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
