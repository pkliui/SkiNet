#!/bin/bash
set -euo pipefail

PROJECT_PATH="${PROJECT_PATH:-$PWD}"

if [[ "$PROJECT_PATH" != /* ]]; then
  PROJECT_PATH="$(realpath "$PROJECT_PATH")"
fi

DB_PATH="${PROJECT_PATH}/mlflow.db"
# MLFLOW_HOST_ARTIFACT_PATH may be injected from the host when running inside Docker
# so that artifact URIs in the DB resolve correctly outside the container.
ARTIFACT_PATH="${MLFLOW_HOST_ARTIFACT_PATH:-${PROJECT_PATH}/mlruns}"
LOG_DIR="${PROJECT_PATH}/logs"
MLFLOW_STDOUT_LOG="${LOG_DIR}/mlflow.stdout.log"
MLFLOW_STDERR_LOG="${LOG_DIR}/mlflow.stderr.log"

BACKEND_STORE_URI="sqlite:////${DB_PATH#/}"
DEFAULT_ARTIFACT_ROOT="file://${ARTIFACT_PATH}"

mkdir -p "${LOG_DIR}"

echo "==> MLflow backend: ${BACKEND_STORE_URI}"
echo "==> MLflow artifacts: ${DEFAULT_ARTIFACT_ROOT}"

# Kill any existing MLflow process on port 5000
if lsof -ti:5000 &>/dev/null; then
  echo "==> Port 5000 in use — killing existing process..."
  kill -9 $(lsof -ti:5000) 2>/dev/null || true
  sleep 1
fi

mlflow server \
  --backend-store-uri "${BACKEND_STORE_URI}" \
  --default-artifact-root "${DEFAULT_ARTIFACT_ROOT}" \
  --host 0.0.0.0 \
  --port 5000 \
  --cors-allowed-origins '*' \
  >"${MLFLOW_STDOUT_LOG}" 2>"${MLFLOW_STDERR_LOG}" &

MLFLOW_PID=$!
echo "==> MLflow PID: ${MLFLOW_PID}"
echo "==> MLflow stdout log: ${MLFLOW_STDOUT_LOG}"
echo "==> MLflow stderr log: ${MLFLOW_STDERR_LOG}"

echo "Waiting for MLflow server..."
until python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" 2>/dev/null; do
  # Check the process didn't die
  if ! kill -0 "$MLFLOW_PID" 2>/dev/null; then
    echo "==> ERROR: MLflow process died unexpectedly"
    exit 1
  fi
  sleep 1
done

echo "==> MLflow server is up!"
