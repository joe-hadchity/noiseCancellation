"""
Virtual Microphone GUI Controller
A simple GUI to control the virtual microphone with noise cancellation
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import logging
from pathlib import Path
import json
import sounddevice as sd

# Import our virtual microphone
from advanced_virtual_mic import AdvancedVirtualMicrophone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VirtualMicGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Virtual Microphone with Noise Cancellation")
        self.root.geometry("600x500")
        
        # Virtual microphone instance
        self.vm = None
        self.is_running = False
        
        # Create GUI elements
        self.create_widgets()
        self.load_device_list()
        
    def create_widgets(self):
        """Create GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Virtual Microphone Controller", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Device selection frame
        device_frame = ttk.LabelFrame(main_frame, text="Audio Devices", padding="10")
        device_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Input device
        ttk.Label(device_frame, text="Input Device:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.input_device_var = tk.StringVar()
        self.input_device_combo = ttk.Combobox(device_frame, textvariable=self.input_device_var, 
                                             state="readonly", width=40)
        self.input_device_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        # VB-Cable device
        ttk.Label(device_frame, text="VB-Cable Device:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.vbcable_device_var = tk.StringVar()
        self.vbcable_device_combo = ttk.Combobox(device_frame, textvariable=self.vbcable_device_var,
                                                state="readonly", width=40)
        self.vbcable_device_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        # Refresh devices button
        refresh_btn = ttk.Button(device_frame, text="Refresh Devices", command=self.load_device_list)
        refresh_btn.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Sample rate
        ttk.Label(settings_frame, text="Sample Rate:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.sample_rate_var = tk.StringVar(value="44100")
        sample_rate_combo = ttk.Combobox(settings_frame, textvariable=self.sample_rate_var,
                                       values=["22050", "44100", "48000"], state="readonly", width=15)
        sample_rate_combo.grid(row=0, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        
        # Chunk size
        ttk.Label(settings_frame, text="Chunk Size:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.chunk_size_var = tk.StringVar(value="1024")
        chunk_size_combo = ttk.Combobox(settings_frame, textvariable=self.chunk_size_var,
                                      values=["512", "1024", "2048", "4096"], state="readonly", width=15)
        chunk_size_combo.grid(row=1, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        
        # Noise reduction strength
        ttk.Label(settings_frame, text="Noise Reduction:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.noise_reduction_var = tk.DoubleVar(value=0.5)
        noise_reduction_scale = ttk.Scale(settings_frame, from_=0.0, to=1.0, 
                                        variable=self.noise_reduction_var, orient=tk.HORIZONTAL)
        noise_reduction_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        # Noise reduction value label
        self.noise_reduction_label = ttk.Label(settings_frame, text="0.5")
        self.noise_reduction_label.grid(row=2, column=2, sticky=tk.W, pady=5, padx=(5, 0))
        
        # Update label when scale changes
        noise_reduction_scale.configure(command=self.update_noise_reduction_label)
        
        # Control frame
        control_frame = ttk.LabelFrame(main_frame, text="Control", padding="10")
        control_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Start/Stop button
        self.start_stop_btn = ttk.Button(control_frame, text="Start Virtual Microphone", 
                                        command=self.toggle_virtual_mic)
        self.start_stop_btn.grid(row=0, column=0, pady=10, padx=5)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Status: Stopped", foreground="red")
        self.status_label.grid(row=0, column=1, pady=10, padx=20)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(main_frame, text="Statistics", padding="10")
        stats_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Statistics labels
        self.stats_labels = {}
        stats_info = [
            ("Chunks Processed:", "chunks_processed"),
            ("Processing Time:", "processing_time"),
            ("Queue Overflows:", "queue_overflows"),
            ("Errors:", "errors")
        ]
        
        for i, (label_text, key) in enumerate(stats_info):
            ttk.Label(stats_frame, text=label_text).grid(row=i, column=0, sticky=tk.W, pady=2)
            self.stats_labels[key] = ttk.Label(stats_frame, text="0")
            self.stats_labels[key].grid(row=i, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        
        # Update statistics button
        update_stats_btn = ttk.Button(stats_frame, text="Update Statistics", 
                                     command=self.update_statistics)
        update_stats_btn.grid(row=len(stats_info), column=0, columnspan=2, pady=10)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        device_frame.columnconfigure(1, weight=1)
        settings_frame.columnconfigure(1, weight=1)
        
    def load_device_list(self):
        """Load available audio devices"""
        try:
            devices = sd.query_devices()
            
            input_devices = []
            vbcable_devices = []
            
            for i, device in enumerate(devices):
                device_name = f"{i}: {device['name']}"
                
                if device['max_input_channels'] > 0:
                    input_devices.append((i, device_name))
                
                if device['max_output_channels'] > 0 and 'cable' in device['name'].lower():
                    vbcable_devices.append((i, device_name))
            
            # Update input device combo
            self.input_device_combo['values'] = [name for _, name in input_devices]
            if input_devices:
                self.input_device_combo.current(0)
            
            # Update VB-Cable device combo
            self.vbcable_device_combo['values'] = [name for _, name in vbcable_devices]
            if vbcable_devices:
                self.vbcable_device_combo.current(0)
            else:
                messagebox.showwarning("VB-Cable Not Found", 
                                     "VB-Cable device not found. Please install VB-Cable first.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load devices: {e}")
    
    def update_noise_reduction_label(self, value):
        """Update noise reduction label"""
        self.noise_reduction_label.config(text=f"{float(value):.2f}")
    
    def get_selected_devices(self):
        """Get selected device indices"""
        try:
            input_device = None
            vbcable_device = None
            
            # Get input device
            input_text = self.input_device_var.get()
            if input_text:
                input_device = int(input_text.split(':')[0])
            
            # Get VB-Cable device
            vbcable_text = self.vbcable_device_var.get()
            if vbcable_text:
                vbcable_device = int(vbcable_text.split(':')[0])
            
            return input_device, vbcable_device
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
            input_device, vbcable_device = self.get_selected_devices()
            
            if vbcable_device is None:
                messagebox.showerror("Error", "Please select a VB-Cable device")
                return
            
            # Create virtual microphone instance
            self.vm = AdvancedVirtualMicrophone(
                input_device=input_device,
                vbcable_device=vbcable_device,
                sample_rate=int(self.sample_rate_var.get()),
                chunk_size=int(self.chunk_size_var.get()),
                noise_reduction_strength=self.noise_reduction_var.get()
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
        self.start_stop_btn.config(text="Stop Virtual Microphone")
        self.status_label.config(text="Status: Running", foreground="green")
        
        # Disable device selection
        self.input_device_combo.config(state="disabled")
        self.vbcable_device_combo.config(state="disabled")
    
    def update_ui_stopped(self):
        """Update UI when stopped"""
        self.start_stop_btn.config(text="Start Virtual Microphone")
        self.status_label.config(text="Status: Stopped", foreground="red")
        
        # Enable device selection
        self.input_device_combo.config(state="readonly")
        self.vbcable_device_combo.config(state="readonly")
    
    def update_statistics(self):
        """Update statistics display"""
        if self.vm and self.is_running:
            stats = self.vm.stats
            self.stats_labels['chunks_processed'].config(text=str(stats['chunks_processed']))
            self.stats_labels['processing_time'].config(text=f"{stats['processing_time']:.2f}s")
            self.stats_labels['queue_overflows'].config(text=str(stats['queue_overflows']))
            self.stats_labels['errors'].config(text=str(stats['errors']))
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            self.stop_virtual_mic()
        self.root.destroy()


def main():
    """Main function"""
    root = tk.Tk()
    app = VirtualMicGUI(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start the GUI
    root.mainloop()


if __name__ == "__main__":
    main()


