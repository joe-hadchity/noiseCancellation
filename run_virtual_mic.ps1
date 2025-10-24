# Virtual Microphone Setup and Launcher (PowerShell)
# ================================================

Write-Host "Virtual Microphone Setup and Launcher" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ and try again" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if required packages are installed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
try {
    python -c "import sounddevice, librosa, noisereduce, numpy" 2>$null
    Write-Host "Dependencies OK" -ForegroundColor Green
} catch {
    Write-Host "Installing required packages..." -ForegroundColor Yellow
    pip install sounddevice librosa noisereduce numpy tensorflow
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to install packages" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Check if VB-Cable is installed
Write-Host "Checking for VB-Cable..." -ForegroundColor Yellow
try {
    python -c "import sounddevice; devices = sounddevice.query_devices(); cable_found = any('cable' in device['name'].lower() for device in devices); exit(0 if cable_found else 1)" 2>$null
    Write-Host "VB-Cable found" -ForegroundColor Green
} catch {
    Write-Host "VB-Cable not found. Setting up VB-Cable..." -ForegroundColor Yellow
    python vbcable_setup.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to setup VB-Cable" -ForegroundColor Red
        Write-Host "Please manually install VB-Cable from https://vb-audio.com/Cable/" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "Please restart your computer for VB-Cable to take effect." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 0
}

# Launch the virtual microphone GUI
Write-Host "Starting Virtual Microphone GUI..." -ForegroundColor Green
python vm_gui.py

Read-Host "Press Enter to exit"


