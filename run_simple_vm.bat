@echo off
echo Virtual Microphone - No Administrator Required
echo ===============================================

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

REM Launch the simple virtual microphone GUI
echo Starting Virtual Microphone GUI...
echo No administrator privileges required!
python simple_vm_gui.py

pause
