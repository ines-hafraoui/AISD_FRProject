# Use Python 3.10 slim for maximum compatibility
FROM --platform=$BUILDPLATFORM python:3.10-slim

# Prevent Python from writing pyc files and enable unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_USE_LEGACY_KERAS=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies required by OpenCV and InsightFace
RUN apt-get update && apt-get install -y \
        build-essential \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        wget \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --progress-bar=off --verbose -r requirements.txt

# Pre-download and cache InsightFace models to avoid runtime delays
RUN python3 - <<'EOF'
import os, traceback
os.environ['TF_USE_LEGACY_KERAS'] = '1'

print("Pre-loading InsightFace models (CPU mode)...")
try:
    from insightface.app import FaceAnalysis
    embedder = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    embedder.prepare(ctx_id=-1, det_size=(640, 640))
    _ = embedder.models.get("recognition", None)
    if _ is None:
        print("Warning: recognition model not found in embedder.")
    else:
        print("Model successfully loaded and cached.")
except Exception as e:
    traceback.print_exc()
EOF

# Copy application code
COPY main.py util.py ./

# Expose FastAPI port
EXPOSE 5000

# Healthcheck for Docker
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:5000/docs')" || exit 1

# Command to run the API
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]