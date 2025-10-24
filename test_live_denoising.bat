@echo off
echo Live Noise Cancellation Test
echo ============================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Checking for VB-Cable installation...
python -c "import sounddevice as sd; devices = sd.query_devices(); cable_devices = [d for d in devices if 'cable' in d['name'].lower()]; print('VB-Cable devices found:' if cable_devices else 'No VB-Cable devices found'); [print(f'  {i}: {d[\"name\"]}') for i, d in enumerate(devices) if 'cable' in d['name'].lower()]"

echo.
echo Starting live noise cancellation test...
echo.
echo Instructions:
echo 1. Make some noise (clap, talk, play music)
echo 2. The system will record for 5 seconds
echo 3. It will process the audio and save the cleaned version
echo 4. You can compare the original vs cleaned audio
echo.

python livetest.py

echo.
echo Test completed! Check the output files:
echo - clean.wav (denoised audio)
echo.
pause

