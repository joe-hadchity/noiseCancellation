# Virtual Microphone with Noise Cancellation

This project creates a virtual microphone that applies real-time noise cancellation to your audio input, making it available to other applications through VB-Cable.

## Features

- **Real-time Noise Cancellation**: Uses machine learning to identify and reduce noise in audio
- **Virtual Audio Output**: Outputs cleaned audio through VB-Cable for use by other applications
- **GUI Controller**: Easy-to-use interface for controlling the virtual microphone
- **Multiple Noise Types**: Recognizes and reduces various types of noise (wind, horns, drilling, etc.)
- **Configurable Settings**: Adjustable noise reduction strength, sample rate, and chunk size
- **Performance Monitoring**: Real-time statistics and performance metrics

## Requirements

- Windows 10/11
- Python 3.8+
- VB-Cable Virtual Audio Cable
- Microphone input device

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Additional packages needed for virtual microphone:
```bash
pip install sounddevice librosa noisereduce numpy tensorflow tkinter
```

### 2. Install VB-Cable

VB-Cable is a virtual audio cable that allows you to route audio between applications.

**Option A: Automatic Installation**
```bash
python vbcable_setup.py
```

**Option B: Manual Installation**
1. Download VB-Cable from https://vb-audio.com/Cable/
2. Run the installer as administrator
3. Restart your computer

### 3. Verify Installation

Run the setup script to verify everything is working:
```bash
python run_virtual_mic.bat
```

## Usage

### GUI Application (Recommended)

1. Run the GUI application:
   ```bash
   python vm_gui.py
   ```

2. Select your input device (microphone)
3. Select VB-Cable as output device
4. Adjust settings as needed:
   - Sample Rate: 22050, 44100, or 48000 Hz
   - Chunk Size: 512, 1024, 2048, or 4096 samples
   - Noise Reduction: 0.0 (no reduction) to 1.0 (maximum reduction)

5. Click "Start Virtual Microphone"
6. Your cleaned audio will now be available through VB-Cable

### Command Line Usage

```bash
python advanced_virtual_mic.py
```

### Basic Virtual Microphone

```bash
python virtual_mic.py
```

## Configuration

The virtual microphone can be configured through a JSON file (`vm_config.json`):

```json
{
  "input_device": null,
  "vbcable_device": null,
  "sample_rate": 44100,
  "chunk_size": 1024,
  "noise_reduction_strength": 0.5,
  "enable_aggressive_mode": false,
  "enable_echo_cancellation": true,
  "enable_automatic_gain_control": true
}
```

## How It Works

1. **Audio Input**: Captures audio from your microphone
2. **Noise Classification**: Uses machine learning to identify noise types
3. **Reference Matching**: Matches identified noise with reference samples
4. **Noise Reduction**: Applies advanced noise reduction algorithms
5. **Audio Output**: Outputs cleaned audio through VB-Cable

## Using with Other Applications

Once the virtual microphone is running, you can use the cleaned audio in any application:

1. **Zoom/Teams**: Select "CABLE Input" as your microphone
2. **OBS Studio**: Add "CABLE Input" as audio source
3. **Discord**: Select "CABLE Input" as input device
4. **Recording Software**: Use "CABLE Input" as audio source

## Troubleshooting

### VB-Cable Not Found
- Ensure VB-Cable is installed and drivers are loaded
- Restart your computer after installation
- Run as administrator if needed

### No Audio Output
- Check that VB-Cable is selected as output device
- Verify input device is working
- Check audio levels and permissions

### High CPU Usage
- Increase chunk size (e.g., 2048 or 4096)
- Reduce sample rate (e.g., 22050 Hz)
- Lower noise reduction strength

### Poor Audio Quality
- Adjust noise reduction strength
- Check microphone quality
- Ensure proper audio levels

## Performance Tips

1. **Chunk Size**: Larger chunks reduce CPU usage but increase latency
2. **Sample Rate**: Lower rates reduce processing load
3. **Noise Reduction**: Higher values use more CPU but provide better cleaning
4. **Reference Files**: Ensure noise reference files are in the `noise/` directory

## File Structure

```
├── virtual_mic.py              # Basic virtual microphone
├── advanced_virtual_mic.py     # Advanced virtual microphone with GUI
├── vm_gui.py                   # GUI controller
├── vbcable_setup.py            # VB-Cable installation helper
├── run_virtual_mic.bat         # Windows batch launcher
├── run_virtual_mic.ps1         # PowerShell launcher
├── vm_config.json              # Configuration file
├── noise/                      # Reference noise samples
│   ├── horn1.wav
│   ├── horn2.wav
│   └── ...
└── backend/                    # Existing noise cancellation backend
    ├── audio.py
    ├── main.py
    └── ...
```

## Advanced Usage

### Custom Noise References

Add your own noise reference files to the `noise/` directory:
- `horn1.wav`, `horn2.wav` - Horn sounds
- `drill1.wav`, `drill2.wav` - Drilling sounds
- `engine1.wav`, `engine2.wav` - Engine sounds
- etc.

### Integration with Existing Backend

The virtual microphone integrates with your existing FastAPI backend:
- Uses the same noise classification model
- Leverages existing denoising functions
- Maintains compatibility with your web interface

### Real-time Processing

The system processes audio in real-time with minimal latency:
- Input callback captures audio chunks
- Processing thread applies noise reduction
- Output callback streams cleaned audio

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure VB-Cable is properly installed
4. Check audio device permissions

## License

This project extends your existing noise cancellation system with virtual microphone capabilities.


