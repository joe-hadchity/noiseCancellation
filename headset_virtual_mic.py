"""
Headset Virtual Microphone - Optimized for headset users
This version works directly with headsets without complex VB-Cable routing
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import logging
from pathlib import Path
import sounddevice as sd
import numpy as np
import wave
import time

# Import our improved virtual microphone
from improved_virtual_mic import ImprovedVirtualMicrophone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeadsetVirtualMicGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Headset Virtual Microphone")
        self.root.geometry("600x700")
        
        # Virtual microphone instance
        self.vm = None
        self.is_running = False
        
        # Create GUI elements
        self.create_widgets()
        self.load_device_list()
        
    def create_widgets(self):
        """Create GUI widgets optimized for headset users"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Headset Virtual Microphone", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Info frame
        info_frame = ttk.LabelFrame(main_frame, text="Headset Setup", padding="10")
        info_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        info_text = """This setup is optimized for headset users.

Setup Options:
1. DIRECT MODE: Process headset mic → Save cleaned audio to file
2. MONITOR MODE: Process headset mic → Play through headset speakers
3. TEAMS MODE: Process headset mic → Route through VB-Cable to Teams

Choose the mode that works best for your setup."""
        
        info_label = ttk.Label(info_frame, text=info_text, wraplength=550, justify=tk.LEFT)
        info_label.grid(row=0, column=0, columnspan=2, pady=5)
        
        # Mode selection
        mode_frame = ttk.LabelFrame(main_frame, text="Processing Mode", padding="10")
        mode_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.mode_var = tk.StringVar(value="direct")
        
        ttk.Radiobutton(mode_frame, text="Direct Mode (Save to file)", 
                       variable=self.mode_var, value="direct").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Radiobutton(mode_frame, text="Monitor Mode (Play through headset)", 
                       variable=self.mode_var, value="monitor").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Radiobutton(mode_frame, text="Teams Mode (Route through VB-Cable)", 
                       variable=self.mode_var, value="teams").grid(row=2, column=0, sticky=tk.W, pady=2)
        
        # Device selection frame
        device_frame = ttk.LabelFrame(main_frame, text="Audio Devices", padding="10")
        device_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Input device (headset mic)
        ttk.Label(device_frame, text="Headset Microphone:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.input_device_var = tk.StringVar()
        self.input_device_combo = ttk.Combobox(device_frame, textvariable=self.input_device_var, 
                                             state="readonly", width=50)
        self.input_device_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        # Output device (headset speakers or VB-Cable)
        ttk.Label(device_frame, text="Output Device:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.output_device_var = tk.StringVar()
        self.output_device_combo = ttk.Combobox(device_frame, textvariable=self.output_device_var,
                                               state="readonly", width=50)
        self.output_device_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        # Refresh devices button
        refresh_btn = ttk.Button(device_frame, text="Refresh Devices", command=self.load_device_list)
        refresh_btn.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Sample rate
        ttk.Label(settings_frame, text="Sample Rate:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.sample_rate_var = tk.StringVar(value="22050")
        sample_rate_combo = ttk.Combobox(settings_frame, textvariable=self.sample_rate_var,
                                       values=["16000", "22050", "44100"], state="readonly", width=15)
        sample_rate_combo.grid(row=0, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        
        # Chunk size
        ttk.Label(settings_frame, text="Chunk Size:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.chunk_size_var = tk.StringVar(value="1024")
        chunk_size_combo = ttk.Combobox(settings_frame, textvariable=self.chunk_size_var,
                                      values=["512", "1024", "2048"], state="readonly", width=15)
        chunk_size_combo.grid(row=1, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        
        # Noise reduction strength
        ttk.Label(settings_frame, text="Noise Reduction:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.noise_reduction_var = tk.DoubleVar(value=0.7)
        noise_reduction_scale = ttk.Scale(settings_frame, from_=0.0, to=1.0, 
                                        variable=self.noise_reduction_var, orient=tk.HORIZONTAL)
        noise_reduction_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        # Noise reduction value label
        self.noise_reduction_label = ttk.Label(settings_frame, text="0.7")
        self.noise_reduction_label.grid(row=2, column=2, sticky=tk.W, pady=5, padx=(5, 0))
        
        # Update label when scale changes
        noise_reduction_scale.configure(command=self.update_noise_reduction_label)
        
        # Output file
        ttk.Label(settings_frame, text="Output File:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.output_file_var = tk.StringVar(value="headset_cleaned_audio.wav")
        output_file_entry = ttk.Entry(settings_frame, textvariable=self.output_file_var, width=40)
        output_file_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        # Control frame
        control_frame = ttk.LabelFrame(main_frame, text="Control", padding="10")
        control_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Start/Stop button
        self.start_stop_btn = ttk.Button(control_frame, text="Start Headset Virtual Microphone", 
                                        command=self.toggle_virtual_mic)
        self.start_stop_btn.grid(row=0, column=0, pady=10, padx=5)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Status: Stopped", foreground="red")
        self.status_label.grid(row=0, column=1, pady=10, padx=20)
        
        # Instructions frame
        instructions_frame = ttk.LabelFrame(main_frame, text="Instructions", padding="10")
        instructions_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        instructions_text = """For Teams/Zoom:
1. Start the virtual microphone
2. In Teams: Go to Settings → Devices
3. Select "CABLE Output (VB-Audio Virtual Cable)" as microphone
4. Select your headset as speaker/headphone

The AI will clean your headset microphone audio in real-time!"""
        
        instructions_label = ttk.Label(instructions_frame, text=instructions_text, wraplength=550, justify=tk.LEFT)
        instructions_label.grid(row=0, column=0, columnspan=2, pady=5)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        device_frame.columnconfigure(1, weight=1)
        settings_frame.columnconfigure(1, weight=1)
        
    def load_device_list(self):
        """Load available audio devices, prioritizing headset devices"""
        try:
            devices = sd.query_devices()
            
            input_devices = []
            output_devices = []
            
            for i, device in enumerate(devices):
                device_name = f"{i}: {device['name']}"
                
                if device['max_input_channels'] > 0:
                    input_devices.append((i, device_name))
                
                if device['max_output_channels'] > 0:
                    output_devices.append((i, device_name))
            
            # Update input device combo (prioritize headset devices)
            headset_inputs = [name for _, name in input_devices if 'headset' in name.lower()]
            other_inputs = [name for _, name in input_devices if 'headset' not in name.lower()]
            all_inputs = headset_inputs + other_inputs
            
            self.input_device_combo['values'] = all_inputs
            if all_inputs:
                self.input_device_combo.current(0)
            
            # Update output device combo
            self.output_device_combo['values'] = [name for _, name in output_devices]
            if output_devices:
                self.output_device_combo.current(0)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load devices: {e}")
    
    def update_noise_reduction_label(self, value):
        """Update noise reduction label"""
        self.noise_reduction_label.config(text=f"{float(value):.2f}")
    
    def get_selected_devices(self):
        """Get selected device indices"""
        try:
            input_device = None
            output_device = None
            
            # Get input device
            input_text = self.input_device_var.get()
            if input_text:
                input_device = int(input_text.split(':')[0])
            
            # Get output device based on mode
            mode = self.mode_var.get()
            if mode == "teams":
                # For Teams mode, try to find VB-Cable output
                output_text = self.output_device_var.get()
                if output_text and 'cable' in output_text.lower():
                    output_device = int(output_text.split(':')[0])
            elif mode == "monitor":
                # For monitor mode, use selected output device
                output_text = self.output_device_var.get()
                if output_text:
                    output_device = int(output_text.split(':')[0])
            # For direct mode, no output device needed
            
            return input_device, output_device
        except Exception as e:
            logger.error(f"Error getting selected devices: {e}")
            return None, None
    
    def toggle_virtual_mic(self):
        """Toggle virtual microphone on/off"""
        if not self.is_running:
            self.start_virtual_mic()
        else:
            self.stop_virtual_mic()
    
    def start_virtual_mic(self):
        """Start the virtual microphone"""
        try:
            input_device, output_device = self.get_selected_devices()
            
            if input_device is None:
                messagebox.showerror("Error", "Please select a headset microphone")
                return
            
            # Create virtual microphone instance
            self.vm = ImprovedVirtualMicrophone(
                input_device=input_device,
                output_device=output_device,
                sample_rate=int(self.sample_rate_var.get()),
                chunk_size=int(self.chunk_size_var.get()),
                noise_reduction_strength=self.noise_reduction_var.get(),
                model_path="final_model.pt",
                output_file=self.output_file_var.get()
            )
            
            # Start in separate thread
            def start_thread():
                try:
                    self.vm.start()
                    self.is_running = True
                    self.root.after(0, self.update_ui_running)
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to start: {e}"))
            
            threading.Thread(target=start_thread, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start virtual microphone: {e}")
    
    def stop_virtual_mic(self):
        """Stop the virtual microphone"""
        try:
            if self.vm:
                self.vm.stop()
                self.vm = None
            
            self.is_running = False
            self.update_ui_stopped()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop virtual microphone: {e}")
    
    def update_ui_running(self):
        """Update UI when running"""
        self.start_stop_btn.config(text="Stop Headset Virtual Microphone")
        self.status_label.config(text="Status: Running", foreground="green")
        
        # Disable device selection
        self.input_device_combo.config(state="disabled")
        self.output_device_combo.config(state="disabled")
    
    def update_ui_stopped(self):
        """Update UI when stopped"""
        self.start_stop_btn.config(text="Start Headset Virtual Microphone")
        self.status_label.config(text="Status: Stopped", foreground="red")
        
        # Enable device selection
        self.input_device_combo.config(state="readonly")
        self.output_device_combo.config(state="readonly")
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            self.stop_virtual_mic()
        self.root.destroy()


def main():
    """Main function"""
    root = tk.Tk()
    app = HeadsetVirtualMicGUI(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start the GUI
    root.mainloop()


if __name__ == "__main__":
    main()

