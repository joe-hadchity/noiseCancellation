@echo off
echo VB-Cable Driver Installation (Administrator Required)
echo ===================================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Running as administrator - proceeding with installation...
    echo.
) else (
    echo This script requires administrator privileges.
    echo Please right-click and select "Run as administrator"
    echo.
    pause
    exit /b 1
)

REM Navigate to vb-cable directory
cd /d "%~dp0vb-cable"

REM Check if installer exists
if not exist "VBCABLE_Setup_x64.exe" (
    echo Error: VBCABLE_Setup_x64.exe not found in vb-cable directory
    echo Please ensure the file exists and try again.
    pause
    exit /b 1
)

echo Installing VB-Cable driver...
echo This may take a few moments...
echo.

REM Run the installer silently
VBCABLE_Setup_x64.exe /S

if %errorLevel% == 0 (
    echo.
    echo VB-Cable installed successfully!
    echo.
    echo IMPORTANT: You may need to restart your computer for the driver to take effect.
    echo After restart, you can test the virtual microphone.
    echo.
) else (
    echo.
    echo Installation failed with error code: %errorLevel%
    echo Please try running the installer manually as administrator.
    echo.
)

echo Press any key to continue...
pause >nul

