from flask import Flask, jsonify
from flask_cors import CORS
import subprocess
import os
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Absolute path to the GUI script on your machine
SCRIPT_PATH = Path(r"C:\Users\DanielAlKabbout\Desktop\noiseCancellation-2\simple_vm_gui.py")

@app.route("/run_simple_vm_gui", methods=["POST"])
def run_simple_vm_gui():
    if not SCRIPT_PATH.exists():
        return jsonify({"error": "simple_vm_gui.py not found", "path": str(SCRIPT_PATH)}), 404
    try:
        # On Windows spawn detached process so Flask returns immediately.
        # DETACHED_PROCESS flag ensures the child has no console attached.
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        creationflags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP

        # Use the same Python interpreter as the server if desired:
        python_exe = "python"

        # Start the GUI script in its directory (cwd)
        proc = subprocess.Popen(
            [python_exe, str(SCRIPT_PATH)],
            cwd=str(SCRIPT_PATH.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            close_fds=True
        )
        return jsonify({"started": True, "pid": proc.pid}), 200
    except Exception as e:
        return jsonify({"error": "failed to start script", "detail": str(e)}), 500

if __name__ == "__main__":
    # Run on port 8000 to match the frontend API_BASE default
    app.run(host="0.0.0.0", port=8000)
