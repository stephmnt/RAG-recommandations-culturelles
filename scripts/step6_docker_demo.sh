#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
IMAGE_NAME="${IMAGE_NAME:-puls-events-rag:step6}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

echo "== Step 6 demo bootstrap =="
echo "root: ${ROOT_DIR}"

echo ""
echo "[1/3] Build FAISS index"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/build_index.py" \
  --config "${ROOT_DIR}/configs/indexing.yaml" \
  --input "${ROOT_DIR}/data/processed/events_processed.parquet" \
  --output "${ROOT_DIR}/artifacts/faiss_index"

echo ""
echo "[2/3] Build Docker image ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" "${ROOT_DIR}"

echo ""
echo "[3/3] Run API container on port ${PORT}"
docker run --rm \
  -p "${PORT}:8000" \
  --env-file "${ROOT_DIR}/.env" \
  -e HOST="${HOST}" \
  -e PORT=8000 \
  -v "${ROOT_DIR}/artifacts:/app/artifacts" \
  -v "${ROOT_DIR}/data:/app/data" \
  -v "${ROOT_DIR}/logs:/app/logs" \
  "${IMAGE_NAME}"
