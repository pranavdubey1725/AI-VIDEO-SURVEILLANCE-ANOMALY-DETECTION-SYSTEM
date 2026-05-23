"""
Start both the FastAPI backend and Streamlit frontend in one command.

Usage:
    python run.py

Opens:
    FastAPI  -> http://localhost:8000
    API docs -> http://localhost:8000/docs
    Streamlit-> http://localhost:8501
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path

ROOT = Path(__file__).parent
PYTHON = sys.executable

processes = []

def shutdown(sig, frame):
    print("\nShutting down...")
    for p in processes:
        p.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

print("=" * 55)
print("AI Video Surveillance System")
print("=" * 55)
print()

# Start FastAPI
print("Starting FastAPI backend on http://localhost:8000 ...")
api_proc = subprocess.Popen(
    [PYTHON, "-m", "uvicorn", "api.main:app",
     "--host", "0.0.0.0", "--port", "8000"],
    cwd=ROOT,
)
processes.append(api_proc)

# Give FastAPI a moment to load the pipeline before Streamlit tries to connect
print("Waiting for models to load (this takes ~15 seconds)...")
time.sleep(15)

# Start Streamlit
print("Starting Streamlit UI on http://localhost:8501 ...")
ui_proc = subprocess.Popen(
    [PYTHON, "-m", "streamlit", "run", "ui/app.py",
     "--server.port", "8501", "--server.headless", "true"],
    cwd=ROOT,
)
processes.append(ui_proc)

print()
print("=" * 55)
print("Both servers running.")
print(f"  FastAPI  : http://localhost:8000")
print(f"  API docs : http://localhost:8000/docs")
print(f"  Streamlit: http://localhost:8501")
print()
print("Press Ctrl+C to stop.")
print("=" * 55)

# Wait for either process to exit
while True:
    for p in processes:
        if p.poll() is not None:
            print(f"Process {p.pid} exited. Shutting down...")
            shutdown(None, None)
    time.sleep(1)
