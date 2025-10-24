@echo off
echo Virtual Microphone Setup and Launcher
echo =====================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if required packages are installed
echo Checking dependencies...
python -c "import sounddevice, librosa, noisereduce, numpy" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install sounddevice librosa noisereduce numpy tensorflow
    if errorlevel 1 (
        echo Error: Failed to install packages
        pause
        exit /b 1
    )
)

REM Check if VB-Cable is installed
echo Checking for VB-Cable...
python -c "import sounddevice; devices = sounddevice.query_devices(); cable_found = any('cable' in device['name'].lower() for device in devices); exit(0 if cable_found else 1)" >nul 2>&1
if errorlevel 1 (
    echo VB-Cable not found. Setting up VB-Cable...
    python vbcable_setup.py
    if errorlevel 1 (
        echo Error: Failed to setup VB-Cable
        echo Please manually install VB-Cable from https://vb-audio.com/Cable/
        pause
        exit /b 1
    )
    echo Please restart your computer for VB-Cable to take effect.
    pause
    exit /b 0
)

REM Launch the virtual microphone GUI
echo Starting Virtual Microphone GUI...
python vm_gui.py

pause


