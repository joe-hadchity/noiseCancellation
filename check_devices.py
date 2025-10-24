import sounddevice as sd

print("Available audio devices:")
print("=" * 50)

devices = sd.query_devices()
for i, device in enumerate(devices):
    print(f"{i}: {device['name']}")
    print(f"   Input channels: {device['max_input_channels']}")
    print(f"   Output channels: {device['max_output_channels']}")
    print(f"   Host API: {device['hostapi']}")
    print()

# Look specifically for VB-Cable devices
print("VB-Cable devices:")
print("=" * 20)
cable_devices = [d for d in devices if 'cable' in d['name'].lower()]
if cable_devices:
    for i, device in enumerate(devices):
        if 'cable' in device['name'].lower():
            print(f"{i}: {device['name']} (Output device for virtual microphone)")
else:
    print("No VB-Cable devices found. You may need to install VB-Cable first.")
