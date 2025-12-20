#!/usr/bin/env bash
set -euo pipefail

TRITON_REPO=${TRITON_REPO:-/app/model_repository}
TRITON_HTTP_PORT=${TRITON_HTTP_PORT:-8000}
TRITON_GRPC_PORT=${TRITON_GRPC_PORT:-8001}
TRITON_METRICS_PORT=${TRITON_METRICS_PORT:-8002}
FASTAPI_PORT=${FASTAPI_PORT:-3000}

echo "[start] Using model repository: ${TRITON_REPO}"

START_TRITON=true
if [ ! -f "${TRITON_REPO}/fr_model/1/model.onnx" ]; then
  echo "[start] Models missing. Downloading Buffalo_L pack..."
  wget -q https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
  unzip -q buffalo_l.zip -d temp_pack
  
  mkdir -p "${TRITON_REPO}/face_detector/1" "${TRITON_REPO}/fr_model/1"
  
  # det_10g = Detector, w600k_r50 = Recognition
  cp temp_pack/det_10g.onnx "${TRITON_REPO}/face_detector/1/model.onnx"
  cp temp_pack/w600k_r50.onnx "${TRITON_REPO}/fr_model/1/model.onnx"
  
  rm -rf buffalo_l.zip temp_pack
  echo "[start] Models prepared successfully."
fi

if [ "${START_TRITON}" = true ]; then
  tritonserver --model-repository="${TRITON_REPO}" \
    --http-port="${TRITON_HTTP_PORT}" \
    --grpc-port="${TRITON_GRPC_PORT}" \
    --metrics-port="${TRITON_METRICS_PORT}" &
  TRITON_PID=$!

  cleanup() {
    echo "[start] Stopping Triton (pid=${TRITON_PID})"
    kill "${TRITON_PID}" 2>/dev/null || true
  }
  trap cleanup EXIT
fi

echo "[start] Launching FastAPI on port ${FASTAPI_PORT}"
uvicorn app:app --host 0.0.0.0 --port "${FASTAPI_PORT}"
