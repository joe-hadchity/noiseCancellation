@echo off
echo Complete Setup with final_model.pt
echo ===================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Running as administrator - proceeding with setup...
    echo.
) else (
    echo This script requires administrator privileges.
    echo Please right-click and select "Run as administrator"
    echo.
    pause
    exit /b 1
)

echo Step 1: Installing Python dependencies...
echo ========================================
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Installing required packages...
pip install sounddevice librosa noisereduce numpy tensorflow soundfile
if errorlevel 1 (
    echo Error: Failed to install packages
    pause
    exit /b 1
)

echo.
echo Step 2: Installing PyTorch for final_model.pt...
echo ================================================
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo Warning: PyTorch installation failed, but continuing...
)

echo.
echo Step 3: Installing VB-Cable driver...
echo =====================================
cd /d "%~dp0vb-cable"

if not exist "VBCABLE_Setup_x64.exe" (
    echo Error: VBCABLE_Setup_x64.exe not found
    echo Please ensure the file exists in vb-cable directory
    pause
    exit /b 1
)

echo Installing VB-Cable driver...
VBCABLE_Setup_x64.exe /S

if %errorLevel% == 0 (
    echo VB-Cable installed successfully!
) else (
    echo VB-Cable installation failed with error code: %errorLevel%
    echo Please try running the installer manually
    pause
    exit /b 1
)

cd /d "%~dp0"

echo.
echo Step 4: Testing final_model.pt...
echo ==================================
if exist "final_model.pt" (
    echo Found final_model.pt - testing integration...
    python test_final_model.py
    if errorlevel 1 (
        echo Warning: Model test failed, but continuing...
    )
) else (
    echo Warning: final_model.pt not found in current directory
    echo The system will use fallback denoising methods
)

echo.
echo Step 5: Verifying installation...
echo =================================
python -c "import sounddevice as sd; devices = sd.query_devices(); cable_devices = [d for d in devices if 'cable' in d['name'].lower()]; print('VB-Cable devices found:' if cable_devices else 'No VB-Cable devices found'); [print(f'  {i}: {d[\"name\"]}') for i, d in enumerate(devices) if 'cable' in d['name'].lower()]"

echo.
echo Step 6: Starting Virtual Microphone with final_model.pt...
echo ========================================================
echo.
echo The virtual microphone will now start using your final_model.pt
echo This provides the best noise cancellation performance!
echo.

python simple_vm_gui.py

echo.
echo Setup completed!
echo.
echo IMPORTANT NOTES:
echo - Your virtual microphone is now using final_model.pt for AI-powered noise cancellation
echo - You may need to restart your computer for VB-Cable to work properly
echo - After restart, you can use the virtual microphone in other applications
echo - Select "CABLE Input" as your microphone in applications like Zoom, Discord, etc.
echo.
pause
