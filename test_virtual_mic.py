"""
Test script for Virtual Microphone Integration
Tests the virtual microphone with your existing noise cancellation system
"""

import numpy as np
import sounddevice as sd
import time
import logging
from pathlib import Path

# Import your existing modules
try:
    from backend.audio import predict_noises, denoise_with_refs, CLASSES
    from backend.deps import load_shared_model
    BACKEND_AVAILABLE = True
    print("✓ Backend modules imported successfully")
except ImportError as e:
    BACKEND_AVAILABLE = False
    print(f"✗ Backend modules not available: {e}")

# Import virtual microphone
try:
    from advanced_virtual_mic import AdvancedVirtualMicrophone
    VM_AVAILABLE = True
    print("✓ Virtual microphone module imported successfully")
except ImportError as e:
    VM_AVAILABLE = False
    print(f"✗ Virtual microphone module not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_audio_devices():
    """Test available audio devices"""
    print("\n=== Audio Devices Test ===")
    try:
        devices = sd.query_devices()
        print(f"Found {len(devices)} audio devices:")
        
        input_devices = []
        vbcable_devices = []
        
        for i, device in enumerate(devices):
            print(f"  {i}: {device['name']}")
            print(f"     Input: {device['max_input_channels']}, Output: {device['max_output_channels']}")
            
            if device['max_input_channels'] > 0:
                input_devices.append(i)
            
            if device['max_output_channels'] > 0 and 'cable' in device['name'].lower():
                vbcable_devices.append(i)
        
        print(f"\nInput devices: {input_devices}")
        print(f"VB-Cable devices: {vbcable_devices}")
        
        return input_devices, vbcable_devices
        
    except Exception as e:
        print(f"✗ Error testing audio devices: {e}")
        return [], []

def test_backend_integration():
    """Test backend integration"""
    print("\n=== Backend Integration Test ===")
    
    if not BACKEND_AVAILABLE:
        print("✗ Backend not available")
        return False
    
    try:
        # Test model loading
        model = load_shared_model()
        print("✓ Model loaded successfully")
        
        # Test noise classification
        test_audio = np.random.randn(44100)  # 1 second of random audio
        preds, top_indices = predict_noises(test_audio, 44100)
        print(f"✓ Noise prediction working: {len(top_indices)} top indices")
        
        # Test denoising
        cleaned = denoise_with_refs(test_audio, 44100)
        print(f"✓ Denoising working: output shape {cleaned.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Backend integration test failed: {e}")
        return False

def test_virtual_microphone():
    """Test virtual microphone functionality"""
    print("\n=== Virtual Microphone Test ===")
    
    if not VM_AVAILABLE:
        print("✗ Virtual microphone not available")
        return False
    
    try:
        # Create virtual microphone instance
        vm = AdvancedVirtualMicrophone(
            sample_rate=44100,
            chunk_size=1024,
            noise_reduction_strength=0.5
        )
        print("✓ Virtual microphone instance created")
        
        # Test device detection
        input_devices, vbcable_devices = test_audio_devices()
        
        if not vbcable_devices:
            print("⚠ VB-Cable device not found - install VB-Cable for full functionality")
            return False
        
        print("✓ VB-Cable device found")
        return True
        
    except Exception as e:
        print(f"✗ Virtual microphone test failed: {e}")
        return False

def test_noise_references():
    """Test noise reference files"""
    print("\n=== Noise References Test ===")
    
    noise_dir = Path("noise")
    if not noise_dir.exists():
        print("✗ Noise directory not found")
        return False
    
    noise_files = list(noise_dir.glob("*.wav"))
    print(f"Found {len(noise_files)} noise reference files:")
    
    for file in noise_files:
        print(f"  - {file.name}")
    
    if len(noise_files) == 0:
        print("⚠ No noise reference files found")
        return False
    
    print("✓ Noise reference files available")
    return True

def test_real_time_processing():
    """Test real-time audio processing"""
    print("\n=== Real-time Processing Test ===")
    
    if not VM_AVAILABLE:
        print("✗ Virtual microphone not available")
        return False
    
    try:
        vm = AdvancedVirtualMicrophone(
            sample_rate=22050,  # Lower sample rate for testing
            chunk_size=512,
            noise_reduction_strength=0.3
        )
        
        # Test audio processing without starting streams
        test_audio = np.random.randn(22050)  # 1 second of test audio
        cleaned_audio = vm.denoise_audio(test_audio)
        
        print(f"✓ Real-time processing test passed")
        print(f"  Input shape: {test_audio.shape}")
        print(f"  Output shape: {cleaned_audio.shape}")
        print(f"  Processing time: {vm.stats['processing_time']:.4f}s")
        
        return True
        
    except Exception as e:
        print(f"✗ Real-time processing test failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("Virtual Microphone Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Audio Devices", test_audio_devices),
        ("Backend Integration", test_backend_integration),
        ("Virtual Microphone", test_virtual_microphone),
        ("Noise References", test_noise_references),
        ("Real-time Processing", test_real_time_processing)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Virtual microphone is ready to use.")
        print("\nNext steps:")
        print("1. Run: python vm_gui.py")
        print("2. Select your microphone and VB-Cable device")
        print("3. Click 'Start Virtual Microphone'")
    else:
        print("\n⚠ Some tests failed. Please check the issues above.")
        print("\nCommon solutions:")
        print("- Install VB-Cable: python vbcable_setup.py")
        print("- Install dependencies: pip install -r requirements.txt")
        print("- Check audio device permissions")

if __name__ == "__main__":
    run_comprehensive_test()


