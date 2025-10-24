@echo off
echo Headset Virtual Microphone
echo ==========================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Starting Headset Virtual Microphone...
echo This version is optimized for headset users!
echo.

python headset_virtual_mic.py

pause

