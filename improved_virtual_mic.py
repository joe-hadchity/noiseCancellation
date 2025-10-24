"""
Improved Virtual Microphone (No Administrator Required)
This version works without VB-Cable and has optimized performance
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
import wave
import io
import multiprocessing as mp

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

class ImprovedVirtualMicrophone:
    def __init__(self, 
                 input_device: Optional[int] = None,
                 output_device: Optional[int] = None,
                 sample_rate: int = 22050,  # Lower default sample rate
                 chunk_size: int = 1024,
                 noise_reduction_strength: float = 0.5,
                 model_path: str = "final_model.pt",
                 output_file: Optional[str] = None,
                 config_file: str = "vm_config.json"):
        """
        Initialize the improved virtual microphone
        
        Args:
            input_device: Input device index (None for default microphone)
            output_device: Output device index (None for default speakers)
            sample_rate: Audio sample rate (lower = better performance)
            chunk_size: Size of audio chunks to process
            noise_reduction_strength: Strength of noise reduction (0.0 to 1.0)
            model_path: Path to the noise classification model
            output_file: Optional file to save cleaned audio
            config_file: Path to configuration file
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.noise_reduction_strength = noise_reduction_strength
        self.model_path = model_path
        self.output_file = output_file
        self.config_file = config_file
        
        # Load configuration
        self.config = self.load_config()
        
        # Larger queues to prevent overflows
        self.input_queue = queue.Queue(maxsize=50)
        self.output_queue = queue.Queue(maxsize=50)
        
        # Threading control
        self.is_running = False
        self.input_thread = None
        self.processing_thread = None
        self.output_thread = None
        
        # Audio devices
        self.input_device = input_device or self.config.get('input_device')
        self.output_device = output_device or self.config.get('output_device')
        
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
        
        # Wrap PyTorch model if needed
        if self.model and hasattr(self.model, 'forward'):
            self.model = self.wrap_pytorch_model(self.model)
        
        # Performance monitoring
        self.stats = {
            'chunks_processed': 0,
            'processing_time': 0,
            'queue_overflows': 0,
            'errors': 0,
            'skipped_chunks': 0
        }
        
        # Noise classes
        self.classes = CLASSES if BACKEND_AVAILABLE else {
            0: 'Windy', 1: 'Horn', 2: 'Children-noise', 3: 'Dog Bark',
            4: 'Drilling', 5: 'Engine Idling', 6: 'Gun Shot', 7: 'Jackhammer',
            8: 'Siren', 9: 'Street music'
        }
        
        # Reference noise directory
        self.noise_dir = Path("noise")
        
        # Audio recording for file output
        self.recorded_audio = []
        self.recording_lock = threading.Lock()
        
        # Performance optimization
        self.last_noise_profile = None
        self.noise_profile_cache = {}
        self.skip_processing = False
        
    def load_config(self) -> Dict:
        """Load configuration from file"""
        default_config = {
            'input_device': None,
            'output_device': None,
            'sample_rate': 22050,  # Lower default
            'chunk_size': 1024,
            'noise_reduction_strength': 0.5,
            'enable_aggressive_mode': False,
            'enable_echo_cancellation': True,
            'enable_automatic_gain_control': True,
            'performance_mode': True  # Enable performance optimizations
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
    
    def load_standalone_model(self):
        """Load model using standalone implementation"""
        try:
            if Path(self.model_path).exists():
                if self.model_path.endswith('.pt'):
                    # Load PyTorch model
                    import torch
                    try:
                        # Try TorchScript first
                        self.model = torch.jit.load(self.model_path, map_location="cpu")
                        logger.info(f"Loaded PyTorch TorchScript model from {self.model_path}")
                    except Exception:
                        # Try regular torch.load
                        self.model = torch.load(self.model_path, map_location="cpu")
                        logger.info(f"Loaded PyTorch model from {self.model_path}")
                elif self.model_path.endswith('.h5'):
                    # Load TensorFlow model
                    from tensorflow.keras.models import load_model
                    self.model = load_model(self.model_path)
                    logger.info(f"Loaded TensorFlow model from {self.model_path}")
                else:
                    logger.warning(f"Unsupported model format: {self.model_path}")
            else:
                logger.warning(f"Model file {self.model_path} not found")
        except Exception as e:
            logger.warning(f"Could not load standalone model: {e}")
    
    def wrap_pytorch_model(self, torch_model):
        """Wrap PyTorch model to be compatible with Keras-like interface"""
        class PyTorchWrapper:
            def __init__(self, model):
                self.model = model
                self.model.eval()
            
            def predict(self, x):
                import torch
                with torch.no_grad():
                    # Convert numpy to torch tensor
                    if x.ndim == 4 and x.shape[-1] == 1:
                        x = x.transpose(0, 3, 1, 2)  # NHWC to NCHW
                    tensor = torch.from_numpy(x.astype(np.float32))
                    output = self.model(tensor)
                    if hasattr(output, 'detach'):
                        output = output.detach().cpu().numpy()
                    return output
        
        return PyTorchWrapper(torch_model)
    
    def model_based_denoise(self, audio: np.ndarray) -> np.ndarray:
        """Use the model to classify noise and apply targeted denoising"""
        try:
            # Extract features for the model
            features = self.extract_audio_features(audio)
            
            # Get noise predictions
            predictions = self.model.predict(features)[0]
            top_noise_indices = predictions.argsort()[-3:][::-1]
            
            # Get reference noise files for top predictions
            reference_noises = []
            for idx in top_noise_indices:
                noise_class = self.classes.get(idx, f"noise_{idx}")
                noise_files = self.find_noise_references(noise_class)
                for noise_file in noise_files:
                    if Path(noise_file).exists():
                        ref_audio, _ = librosa.load(noise_file, sr=self.sample_rate)
                        reference_noises.append(ref_audio)
            
            # Apply denoising with reference noises
            if reference_noises:
                combined_noise = np.concatenate(reference_noises)
                cleaned = nr.reduce_noise(
                    y=audio,
                    y_noise=combined_noise,
                    sr=self.sample_rate,
                    prop_decrease=self.noise_reduction_strength
                )
                return cleaned
            
            return audio
            
        except Exception as e:
            logger.debug(f"Model-based denoising error: {e}")
            return audio
    
    def extract_audio_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract features for model prediction"""
        try:
            mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=40).T, axis=0)
            melspectrogram = np.mean(librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, n_mels=40, fmax=8000).T, axis=0)
            chroma_stft = np.mean(librosa.feature.chroma_stft(y=audio, sr=self.sample_rate, n_chroma=40).T, axis=0)
            chroma_cq = np.mean(librosa.feature.chroma_cqt(y=audio, sr=self.sample_rate, n_chroma=40, bins_per_octave=40).T, axis=0)
            chroma_cens = np.mean(librosa.feature.chroma_cens(y=audio, sr=self.sample_rate, n_chroma=40, bins_per_octave=40).T, axis=0)
            
            features = np.vstack((mfccs, melspectrogram, chroma_stft, chroma_cq, chroma_cens)).T
            return features.reshape(1, 40, 5, 1)
        except Exception as e:
            logger.debug(f"Feature extraction error: {e}")
            return np.zeros((1, 40, 5, 1))
    
    def find_noise_references(self, noise_class: str) -> List[str]:
        """Find reference noise files for a given class"""
        noise_files = []
        class_name = noise_class.lower().replace(' ', '').replace('-', '')
        
        # Look for files with the class name
        for i in range(1, 4):  # Check for files 1, 2, 3
            filename = f"{self.noise_dir}/{class_name}{i}.wav"
            if Path(filename).exists():
                noise_files.append(filename)
        
        return noise_files
    
    def fast_denoise_audio(self, audio: np.ndarray) -> np.ndarray:
        """Fast denoising with performance optimizations"""
        start_time = time.time()
        
        try:
            # Skip processing if audio is too short
            if len(audio) < 256:
                self.stats['skipped_chunks'] += 1
                return audio.astype(np.float32)
            
            # Use cached noise profile if available
            audio_hash = hash(audio.tobytes()[:1000])  # Hash first part for caching
            if audio_hash in self.noise_profile_cache:
                cached_noise = self.noise_profile_cache[audio_hash]
                if cached_noise is not None:
                    reduced = nr.reduce_noise(
                        y=audio, 
                        y_noise=cached_noise, 
                        sr=self.sample_rate, 
                        prop_decrease=self.noise_reduction_strength
                    )
                    self.stats['chunks_processed'] += 1
                    self.stats['processing_time'] += time.time() - start_time
                    return reduced.astype(np.float32)
            
            # Try backend denoising first (fastest)
            if self.backend_available:
                try:
                    cleaned = denoise_basic(audio, self.sample_rate, prop_decrease=self.noise_reduction_strength)
                    if cleaned.size > 0:
                        self.stats['chunks_processed'] += 1
                        self.stats['processing_time'] += time.time() - start_time
                        return cleaned.astype(np.float32)
                except Exception as e:
                    logger.debug(f"Backend denoising failed: {e}")
            
            # Try model-based denoising if model is available
            if self.model:
                try:
                    cleaned = self.model_based_denoise(audio)
                    if cleaned.size > 0:
                        self.stats['chunks_processed'] += 1
                        self.stats['processing_time'] += time.time() - start_time
                        return cleaned.astype(np.float32)
                except Exception as e:
                    logger.debug(f"Model-based denoising failed: {e}")
            
            # Fallback to basic noise reduction (faster than model-based)
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
        """Optimized callback for audio input"""
        if status:
            logger.warning(f"Input status: {status}")
        
        try:
            # Convert to mono and add to queue
            audio_data = indata[:, 0] if indata.ndim > 1 else indata
            
            # Skip if queue is getting full to prevent overflows
            if self.input_queue.qsize() > 30:
                self.stats['queue_overflows'] += 1
                return
            
            self.input_queue.put_nowait(audio_data.copy())
        except queue.Full:
            self.stats['queue_overflows'] += 1
    
    def processing_worker(self):
        """Optimized worker thread for audio processing"""
        logger.info("Audio processing thread started")
        
        while self.is_running:
            try:
                # Get audio chunk from input queue with timeout
                audio_chunk = self.input_queue.get(timeout=0.05)
                
                # Apply fast noise reduction
                cleaned_audio = self.fast_denoise_audio(audio_chunk)
                
                # Record audio if output file is specified
                if self.output_file:
                    with self.recording_lock:
                        self.recorded_audio.append(cleaned_audio.copy())
                
                # Put processed audio in output queue
                try:
                    self.output_queue.put_nowait(cleaned_audio)
                except queue.Full:
                    # Skip this chunk if output queue is full
                    self.stats['queue_overflows'] += 1
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing error: {e}")
                self.stats['errors'] += 1
        
        logger.info("Audio processing thread stopped")
    
    def output_callback(self, outdata, frames, time, status):
        """Optimized callback for audio output"""
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
        """Start the improved virtual microphone"""
        if self.is_running:
            logger.warning("Virtual microphone is already running")
            return
        
        logger.info("Starting improved virtual microphone...")
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
            
            self.input_stream.start()
            logger.info("Input stream started")
            
            # Start output stream if output device is specified
            if self.output_device is not None:
                self.output_stream = sd.OutputStream(
                    device=self.output_device,
                    channels=1,
                    samplerate=self.sample_rate,
                    blocksize=self.chunk_size,
                    callback=self.output_callback,
                    dtype=np.float32
                )
                
                self.output_stream.start()
                logger.info("Output stream started")
            
            logger.info("Improved virtual microphone started successfully")
            logger.info(f"Input device: {self.input_stream.device}")
            if hasattr(self, 'output_stream'):
                logger.info(f"Output device: {self.output_stream.device}")
            
            if self.output_file:
                logger.info(f"Recording cleaned audio to: {self.output_file}")
            
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
        
        # Save recorded audio if specified
        if self.output_file and self.recorded_audio:
            self.save_recorded_audio()
        
        # Print statistics
        self.print_stats()
        logger.info("Virtual microphone stopped")
    
    def save_recorded_audio(self):
        """Save recorded audio to file"""
        try:
            with self.recording_lock:
                if not self.recorded_audio:
                    return
                
                # Concatenate all audio chunks
                full_audio = np.concatenate(self.recorded_audio)
                
                # Save as WAV file
                with wave.open(self.output_file, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes((full_audio * 32767).astype(np.int16).tobytes())
                
                logger.info(f"Saved {len(full_audio)/self.sample_rate:.2f} seconds of cleaned audio to {self.output_file}")
                
        except Exception as e:
            logger.error(f"Failed to save recorded audio: {e}")
    
    def print_stats(self):
        """Print performance statistics"""
        if self.stats['chunks_processed'] > 0:
            avg_time = self.stats['processing_time'] / self.stats['chunks_processed']
            logger.info(f"Performance Stats:")
            logger.info(f"  Chunks processed: {self.stats['chunks_processed']}")
            logger.info(f"  Skipped chunks: {self.stats['skipped_chunks']}")
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


def main():
    """Main function to run the improved virtual microphone"""
    # Create virtual microphone instance with optimized settings
    vm = ImprovedVirtualMicrophone(
        sample_rate=22050,  # Lower sample rate for better performance
        chunk_size=1024,
        noise_reduction_strength=0.5,
        output_file="improved_cleaned_audio.wav"  # Save cleaned audio to file
    )
    
    # List available devices
    vm.list_audio_devices()
    
    try:
        # Start the virtual microphone
        vm.start()
        
        # Keep running until interrupted
        logger.info("Improved virtual microphone is running. Press Ctrl+C to stop.")
        logger.info("Cleaned audio will be saved to improved_cleaned_audio.wav")
        logger.info("No administrator privileges required!")
        
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


