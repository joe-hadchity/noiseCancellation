"""
Quick Demo - Improved Virtual Microphone
Tests the improved virtual microphone with better performance
"""

import numpy as np
import sounddevice as sd
import time
import logging
from pathlib import Path

# Import our improved virtual microphone
from improved_virtual_mic import ImprovedVirtualMicrophone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_demo():
    """Quick demo of the improved virtual microphone"""
    print("Improved Virtual Microphone - Quick Demo")
    print("=" * 45)
    print("No administrator privileges required!")
    print("No VB-Cable installation needed!")
    print()
    
    # Create virtual microphone instance with optimized settings
    vm = ImprovedVirtualMicrophone(
        sample_rate=22050,  # Lower sample rate for better performance
        chunk_size=1024,
        noise_reduction_strength=0.5,
        output_file="quick_demo_cleaned.wav"
    )
    
    print("Available audio devices:")
    vm.list_audio_devices()
    
    print("\nStarting 3-second demo recording...")
    print("Please speak into your microphone for 3 seconds.")
    print("The cleaned audio will be saved to 'quick_demo_cleaned.wav'")
    print()
    
    try:
        # Start the virtual microphone
        vm.start()
        
        # Record for 3 seconds
        for i in range(3, 0, -1):
            print(f"Recording... {i}")
            time.sleep(1)
        
        # Stop the virtual microphone
        vm.stop()
        
        print("\nDemo completed!")
        print("Check 'quick_demo_cleaned.wav' for the cleaned audio.")
        
        # Show statistics
        print(f"\nPerformance Statistics:")
        print(f"  Chunks processed: {vm.stats['chunks_processed']}")
        print(f"  Skipped chunks: {vm.stats['skipped_chunks']}")
        print(f"  Processing time: {vm.stats['processing_time']:.2f}s")
        print(f"  Queue overflows: {vm.stats['queue_overflows']}")
        print(f"  Errors: {vm.stats['errors']}")
        
        if vm.stats['queue_overflows'] == 0:
            print("\n✓ Excellent performance - no queue overflows!")
        elif vm.stats['queue_overflows'] < 10:
            print("\n✓ Good performance - minimal queue overflows")
        else:
            print("\n⚠ Some performance issues - consider reducing sample rate or chunk size")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        vm.stop()

def test_file_processing():
    """Test processing a file"""
    print("\nFile Processing Test")
    print("=" * 20)
    
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
    print(f"\nTesting improved noise cancellation on: {test_file.name}")
    
    try:
        import librosa
        import soundfile as sf
        
        # Load test audio
        audio, sr = librosa.load(test_file, sr=22050)
        print(f"Loaded audio: {len(audio)/sr:.2f} seconds at {sr} Hz")
        
        # Create virtual microphone for processing
        vm = ImprovedVirtualMicrophone(
            sample_rate=sr,
            chunk_size=1024,
            noise_reduction_strength=0.5
        )
        
        # Process the audio
        print("Processing audio with improved algorithm...")
        cleaned_audio = vm.fast_denoise_audio(audio)
        
        # Save cleaned audio
        output_file = f"improved_cleaned_{test_file.name}"
        sf.write(output_file, cleaned_audio, sr)
        
        print(f"Cleaned audio saved to: {output_file}")
        print(f"Processing statistics:")
        print(f"  Chunks processed: {vm.stats['chunks_processed']}")
        print(f"  Skipped chunks: {vm.stats['skipped_chunks']}")
        print(f"  Processing time: {vm.stats['processing_time']:.2f}s")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")

def main():
    """Main demo function"""
    print("Virtual Microphone - Improved Version Demo")
    print("=" * 50)
    
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
    print("1. Quick live recording demo (3 seconds)")
    print("2. Test file processing demo")
    print("3. Both")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            quick_demo()
        elif choice == "2":
            test_file_processing()
        elif choice == "3":
            quick_demo()
            test_file_processing()
        else:
            print("Invalid choice. Running quick demo...")
            quick_demo()
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")

if __name__ == "__main__":
    main()


