"""
Test script to verify final_model.pt is working with the virtual microphone
"""

import numpy as np
import sounddevice as sd
import librosa
from improved_virtual_mic import ImprovedVirtualMicrophone
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test if final_model.pt loads correctly"""
    print("Testing final_model.pt loading...")
    
    try:
        vm = ImprovedVirtualMicrophone(
            model_path="final_model.pt",
            sample_rate=22050,
            chunk_size=1024
        )
        
        if vm.model:
            print("✅ Model loaded successfully!")
            print(f"Model type: {type(vm.model)}")
            
            # Test model prediction with dummy data
            dummy_features = np.random.randn(1, 40, 5, 1).astype(np.float32)
            try:
                predictions = vm.model.predict(dummy_features)
                print(f"✅ Model prediction successful!")
                print(f"Prediction shape: {predictions.shape}")
                print(f"Sample predictions: {predictions[0][:5]}")
            except Exception as e:
                print(f"❌ Model prediction failed: {e}")
        else:
            print("❌ Model not loaded")
            
    except Exception as e:
        print(f"❌ Failed to create virtual microphone: {e}")

def test_audio_processing():
    """Test audio processing with the model"""
    print("\nTesting audio processing...")
    
    try:
        vm = ImprovedVirtualMicrophone(
            model_path="final_model.pt",
            sample_rate=22050,
            chunk_size=1024,
            noise_reduction_strength=0.5
        )
        
        # Create dummy audio
        duration = 1.0  # 1 second
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))  # 440Hz tone + noise
        
        print(f"Testing with {len(audio)} samples of audio...")
        
        # Test feature extraction
        features = vm.extract_audio_features(audio)
        print(f"✅ Features extracted: {features.shape}")
        
        # Test model-based denoising
        cleaned = vm.model_based_denoise(audio)
        print(f"✅ Audio denoised: {len(cleaned)} samples")
        
        # Test fast denoising
        fast_cleaned = vm.fast_denoise_audio(audio)
        print(f"✅ Fast denoising: {len(fast_cleaned)} samples")
        
    except Exception as e:
        print(f"❌ Audio processing failed: {e}")

def test_live_recording():
    """Test live recording with the model"""
    print("\nTesting live recording (5 seconds)...")
    
    try:
        vm = ImprovedVirtualMicrophone(
            model_path="final_model.pt",
            sample_rate=22050,
            chunk_size=1024,
            output_file="test_final_model_output.wav"
        )
        
        print("Recording 5 seconds of audio...")
        print("Make some noise to test the model!")
        
        # Start the virtual microphone
        vm.start()
        
        # Let it run for 5 seconds
        import time
        time.sleep(5)
        
        # Stop and save
        vm.stop()
        
        print("✅ Live recording test completed!")
        print("Check 'test_final_model_output.wav' for the cleaned audio")
        
    except Exception as e:
        print(f"❌ Live recording failed: {e}")

if __name__ == "__main__":
    print("Final Model.pt Test Suite")
    print("=" * 40)
    
    # Test 1: Model loading
    test_model_loading()
    
    # Test 2: Audio processing
    test_audio_processing()
    
    # Test 3: Live recording (optional)
    response = input("\nTest live recording? (y/n): ").lower().strip()
    if response == 'y':
        test_live_recording()
    
    print("\nTest completed!")
