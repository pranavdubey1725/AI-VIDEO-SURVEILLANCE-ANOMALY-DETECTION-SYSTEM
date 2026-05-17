FROM python:3.11-slim

# OpenCV runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies.
# PyTorch CUDA wheels live on a separate index; --extra-index-url makes pip
# look there in addition to PyPI so the pinned cu124 versions resolve correctly.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu124

COPY . .

# Both ports are declared here; docker-compose maps only the relevant one
# per service, so there is no conflict.
EXPOSE 8000 8501
