"""
Start the surveillance system locally.

Usage:
    python run.py

Opens:
    UI + API -> http://localhost:8000
    API docs -> http://localhost:8000/docs
"""

import subprocess
import sys
import signal
from pathlib import Path

ROOT   = Path(__file__).parent
PYTHON = sys.executable

proc = None

def shutdown(sig, frame):
    print("\nShutting down...")
    if proc:
        proc.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

print("=" * 50)
print("AI Video Surveillance System")
print("=" * 50)
print("  UI + API : http://localhost:8000")
print("  API docs : http://localhost:8000/docs")
print()
print("Press Ctrl+C to stop.")
print("=" * 50)

proc = subprocess.Popen(
    [PYTHON, "-m", "uvicorn", "api.main:app",
     "--host", "0.0.0.0", "--port", "8000"],
    cwd=ROOT,
)

proc.wait()
