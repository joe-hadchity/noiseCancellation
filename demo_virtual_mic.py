"""
Virtual Microphone Demo
Demonstrates the virtual microphone functionality with a short recording
"""

import numpy as np
import sounddevice as sd
import time
import logging
from pathlib import Path

# Import our virtual microphone
from standalone_virtual_mic import StandaloneVirtualMicrophone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_virtual_microphone():
    """Demo the virtual microphone with a short recording"""
    print("Virtual Microphone Demo")
    print("=" * 30)
    
    # Create virtual microphone instance
    vm = StandaloneVirtualMicrophone(
        sample_rate=22050,  # Lower sample rate for demo
        chunk_size=512,
        noise_reduction_strength=0.5,
        output_file="demo_cleaned_audio.wav"
    )
    
    # List available devices
    print("\nAvailable audio devices:")
    vm.list_audio_devices()
    
    print("\nStarting 5-second demo recording...")
    print("Please speak into your microphone for 5 seconds.")
    print("The cleaned audio will be saved to 'demo_cleaned_audio.wav'")
    
    try:
        # Start the virtual microphone
        vm.start()
        
        # Record for 5 seconds
        time.sleep(5)
        
        # Stop the virtual microphone
        vm.stop()
        
        print("\nDemo completed!")
        print("Check 'demo_cleaned_audio.wav' for the cleaned audio.")
        
        # Show statistics
        print(f"\nStatistics:")
        print(f"  Chunks processed: {vm.stats['chunks_processed']}")
        print(f"  Processing time: {vm.stats['processing_time']:.2f}s")
        print(f"  Queue overflows: {vm.stats['queue_overflows']}")
        print(f"  Errors: {vm.stats['errors']}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        vm.stop()

def test_noise_cancellation():
    """Test noise cancellation on a sample audio file"""
    print("\nNoise Cancellation Test")
    print("=" * 25)
    
    # Check if we have test audio files
    test_files = list(Path("test_audio").glob("*.wav")) if Path("test_audio").exists() else []
    
    if not test_files:
        print("No test audio files found in 'test_audio' directory")
        return
    
    print(f"Found {len(test_files)} test audio files:")
    for file in test_files:
        print(f"  - {file.name}")
    
    # Test on first file
    test_file = test_files[0]
    print(f"\nTesting noise cancellation on: {test_file.name}")
    
    try:
        import librosa
        import soundfile as sf
        
        # Load test audio
        audio, sr = librosa.load(test_file, sr=22050)
        print(f"Loaded audio: {len(audio)/sr:.2f} seconds at {sr} Hz")
        
        # Create virtual microphone for processing
        vm = StandaloneVirtualMicrophone(
            sample_rate=sr,
            chunk_size=1024,
            noise_reduction_strength=0.5
        )
        
        # Process the audio
        print("Processing audio...")
        cleaned_audio = vm.denoise_audio(audio)
        
        # Save cleaned audio
        output_file = f"cleaned_{test_file.name}"
        sf.write(output_file, cleaned_audio, sr)
        
        print(f"Cleaned audio saved to: {output_file}")
        print(f"Processing statistics:")
        print(f"  Chunks processed: {vm.stats['chunks_processed']}")
        print(f"  Processing time: {vm.stats['processing_time']:.2f}s")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")

def main():
    """Main demo function"""
    print("Virtual Microphone with Noise Cancellation - Demo")
    print("=" * 55)
    
    # Check if backend is available
    try:
        from backend.audio import CLASSES
        print(f"✓ Backend available with {len(CLASSES)} noise classes")
    except ImportError:
        print("⚠ Backend not available, using standalone implementation")
    
    # Check noise reference files
    noise_dir = Path("noise")
    if noise_dir.exists():
        noise_files = list(noise_dir.glob("*.wav"))
        print(f"✓ Found {len(noise_files)} noise reference files")
    else:
        print("⚠ No noise reference directory found")
    
    print("\nChoose demo option:")
    print("1. Live recording demo (5 seconds)")
    print("2. Test file processing demo")
    print("3. Both")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            demo_virtual_microphone()
        elif choice == "2":
            test_noise_cancellation()
        elif choice == "3":
            demo_virtual_microphone()
            test_noise_cancellation()
        else:
            print("Invalid choice. Running live demo...")
            demo_virtual_microphone()
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")

if __name__ == "__main__":
    main()


