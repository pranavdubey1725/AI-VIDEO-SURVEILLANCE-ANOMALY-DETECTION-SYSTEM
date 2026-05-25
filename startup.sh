#!/bin/bash
set -e

echo "=== AI Video Surveillance System ==="
echo "Starting on port 7860 (FastAPI serves UI + API)..."
exec python -m uvicorn api.main:app --host 0.0.0.0 --port 7860
