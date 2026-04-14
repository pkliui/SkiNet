#!/bin/bash
set -euo pipefail

# Resolve project path: prefer env PROJECT_PATH, otherwise use current working dir
PROJECT_PATH="${PROJECT_PATH:-$PWD}"

# Make absolute if needed
if [[ "$PROJECT_PATH" != /* ]]; then
  PROJECT_PATH="$(realpath "$PROJECT_PATH")"
fi

DB_PATH="${PROJECT_PATH}/mlflow.db"
ARTIFACT_PATH="${PROJECT_PATH}/mlruns"

# Build URIs. Ensure sqlite absolute path uses four slashes: sqlite:////absolute/path
BACKEND_STORE_URI="sqlite:////${DB_PATH#/}"     # remove leading slash before joining
DEFAULT_ARTIFACT_ROOT="file://${ARTIFACT_PATH}"

echo "MLflow backend: ${BACKEND_STORE_URI}"
echo "MLflow artifacts: ${DEFAULT_ARTIFACT_ROOT}"

# Start MLflow server in background
mlflow server \
  --backend-store-uri "${BACKEND_STORE_URI}" \
  --default-artifact-root "${DEFAULT_ARTIFACT_ROOT}" \
  --host 0.0.0.0 \
  --port 5000 &

# Wait for it to be ready
echo "Waiting for MLflow server..."
until python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" 2>/dev/null; do
  sleep 1
done
echo "MLflow server is up!"
