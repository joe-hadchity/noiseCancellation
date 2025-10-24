@echo off
echo Quick Teams Integration Test
echo ============================
echo.

echo This will test if your setup is working for Teams meetings.
echo.

echo Step 1: Checking VB-Cable installation...
python -c "import sounddevice as sd; devices = sd.query_devices(); cable_devices = [d for d in devices if 'cable' in d['name'].lower()]; print('VB-Cable devices found:' if cable_devices else 'No VB-Cable devices found'); [print(f'  {i}: {d[\"name\"]}') for i, d in enumerate(devices) if 'cable' in d['name'].lower()]"

echo.
echo Step 2: Starting virtual microphone for Teams...
echo.
echo IMPORTANT: 
echo - Keep this window open while in your Teams meeting
echo - Your audio will be processed and sent to Teams through VB-Cable
echo - You can adjust settings in the GUI that will open
echo.

echo Starting virtual microphone GUI...
python simple_vm_gui.py

echo.
echo Virtual microphone stopped.
echo.
pause
