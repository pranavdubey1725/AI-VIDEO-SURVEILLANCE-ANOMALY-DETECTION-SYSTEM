#!/bin/bash
set -e

echo "=== AI Video Surveillance System ==="
echo "Starting FastAPI backend on port 8000..."
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &

echo "Waiting for models to load (ResNet50 + LSTM + YOLOv8 on CPU, ~30s)..."
until curl -sf http://localhost:8000/health > /dev/null 2>&1; do
    echo "  Still loading..."
    sleep 5
done
echo "FastAPI ready."

echo "Starting Streamlit on port 7860..."
exec python -m streamlit run ui/app.py \
    --server.port 7860 \
    --server.headless true \
    --server.address 0.0.0.0 \
    --server.enableXsrfProtection false \
    --server.enableCORS false
