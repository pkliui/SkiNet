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

# Migrate schema if the DB already exists (e.g. copied from a previous session).
# The DB may carry an unknown revision stamp (e.g. from a different MLflow install),
# so we stamp it to the current head first, then upgrade.
if [[ -f "${DB_PATH}" ]]; then
  echo "==> Checking MLflow DB schema..."
  EXPECTED_HEAD=$(python - <<'EOF'
from mlflow.store.db.utils import _get_alembic_config
from alembic.script import ScriptDirectory
cfg = _get_alembic_config("sqlite:////dev/null")
script = ScriptDirectory.from_config(cfg)
print(script.get_current_head())
EOF
)
  CURRENT_REV=$(python - "${DB_PATH}" <<'EOF'
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
row = conn.execute("SELECT version_num FROM alembic_version").fetchone()
print(row[0] if row else "none")
EOF
)
  echo "==> DB revision: ${CURRENT_REV}, expected head: ${EXPECTED_HEAD}"
  if [[ "${CURRENT_REV}" != "${EXPECTED_HEAD}" ]]; then
    echo "==> Stamping DB to head revision and upgrading..."
    python - "${DB_PATH}" "${EXPECTED_HEAD}" <<'EOF'
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute("UPDATE alembic_version SET version_num = ?", (sys.argv[2],))
conn.commit()
EOF
    mlflow db upgrade "${BACKEND_STORE_URI}"
    echo "==> Schema upgrade complete."
  else
    echo "==> DB schema is up to date."
  fi
fi

# Kill any existing MLflow process on port 5000
if pkill -9 -f "mlflow server" 2>/dev/null; then
  echo "==> Killed existing mlflow server process"
  sleep 1
fi

setsid mlflow server \
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
