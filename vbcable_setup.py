"""
VB-Cable Virtual Audio Cable Setup and Management
This module helps set up and manage VB-Cable for virtual microphone functionality
"""

import subprocess
import os
import sys
import logging
from pathlib import Path
import requests
import zipfile
import shutil

logger = logging.getLogger(__name__)

class VBCableManager:
    def __init__(self):
        self.vbcable_path = Path("vb-cable")
        self.vbcable_exe = self.vbcable_path / "VBCABLE_Setup_x64.exe"
        self.download_url = "https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack43.zip"
        
    def download_vbcable(self):
        """Download VB-Cable driver"""
        try:
            logger.info("Downloading VB-Cable driver...")
            
            # Create directory
            self.vbcable_path.mkdir(exist_ok=True)
            
            # Download the zip file
            response = requests.get(self.download_url, stream=True)
            response.raise_for_status()
            
            zip_path = self.vbcable_path / "vbcable.zip"
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info("VB-Cable downloaded successfully")
            return zip_path
            
        except Exception as e:
            logger.error(f"Failed to download VB-Cable: {e}")
            return None
    
    def extract_vbcable(self, zip_path):
        """Extract VB-Cable driver"""
        try:
            logger.info("Extracting VB-Cable driver...")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.vbcable_path)
            
            # Clean up zip file
            zip_path.unlink()
            
            logger.info("VB-Cable extracted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract VB-Cable: {e}")
            return False
    
    def install_vbcable(self):
        """Install VB-Cable driver"""
        try:
            if not self.vbcable_exe.exists():
                logger.error("VB-Cable installer not found")
                return False
            
            logger.info("Installing VB-Cable driver...")
            logger.warning("This requires administrator privileges!")
            
            # Run installer
            result = subprocess.run(
                [str(self.vbcable_exe)],
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info("VB-Cable installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install VB-Cable: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Installation error: {e}")
            return False
    
    def setup_vbcable(self):
        """Complete VB-Cable setup process"""
        logger.info("Setting up VB-Cable...")
        
        # Download if not exists
        if not self.vbcable_exe.exists():
            zip_path = self.download_vbcable()
            if not zip_path:
                return False
            
            if not self.extract_vbcable(zip_path):
                return False
        
        # Install
        if not self.install_vbcable():
            return False
        
        logger.info("VB-Cable setup completed successfully!")
        logger.info("Please restart your computer for the driver to take effect.")
        return True
    
    def check_vbcable_installed(self):
        """Check if VB-Cable is installed"""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            
            for device in devices:
                if 'cable' in device['name'].lower():
                    logger.info(f"Found VB-Cable device: {device['name']}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking VB-Cable: {e}")
            return False


def main():
    """Main function for VB-Cable setup"""
    manager = VBCableManager()
    
    if manager.check_vbcable_installed():
        logger.info("VB-Cable is already installed!")
        return
    
    logger.info("VB-Cable not found. Starting setup...")
    
    if manager.setup_vbcable():
        logger.info("Setup completed! Please restart your computer.")
    else:
        logger.error("Setup failed!")


if __name__ == "__main__":
    main()
