@echo off
echo Testing final_model.pt with Virtual Microphone
echo =============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Installing PyTorch if needed...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo Testing final_model.pt integration...
python test_final_model.py

echo.
echo Test completed!
echo.
echo If the test passed, you can now use the virtual microphone with final_model.pt
echo Run: python simple_vm_gui.py
echo.
pause
