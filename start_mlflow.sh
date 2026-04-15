#!/bin/bash
set -euo pipefail

PROJECT_PATH="${PROJECT_PATH:-$PWD}"

if [[ "$PROJECT_PATH" != /* ]]; then
  PROJECT_PATH="$(realpath "$PROJECT_PATH")"
fi

DB_PATH="${PROJECT_PATH}/mlflow.db"
ARTIFACT_PATH="${PROJECT_PATH}/mlruns"

BACKEND_STORE_URI="sqlite:////${DB_PATH#/}"
DEFAULT_ARTIFACT_ROOT="file://${ARTIFACT_PATH}"

echo "MLflow backend: ${BACKEND_STORE_URI}"
echo "MLflow artifacts: ${DEFAULT_ARTIFACT_ROOT}"

# Kill any existing MLflow process on port 5000
if lsof -ti:5000 &>/dev/null; then
  echo "Port 5000 in use — killing existing process..."
  kill -9 $(lsof -ti:5000) 2>/dev/null || true
  sleep 1
fi

mlflow server \
  --backend-store-uri "${BACKEND_STORE_URI}" \
  --default-artifact-root "${DEFAULT_ARTIFACT_ROOT}" \
  --host 0.0.0.0 \
  --port 5000 \
  2>/tmp/mlflow.log &

MLFLOW_PID=$!
echo "MLflow PID: ${MLFLOW_PID}"

echo "Waiting for MLflow server..."
until python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" 2>/dev/null; do
  # Check the process didn't die
  if ! kill -0 "$MLFLOW_PID" 2>/dev/null; then
    echo "ERROR: MLflow process died unexpectedly"
    exit 1
  fi
  sleep 1
done

echo "MLflow server is up!"
