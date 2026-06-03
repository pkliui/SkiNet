#!/usr/bin/env bash
#
# on_start_cpu.sh — SkiNet CPU launcher for Lightning Studio sessions.
#
# Clones/updates the SkiNet repo, pulls the Docker image, launches a container
# and either drops into an interactive shell or runs a test-only pass.
#
# Logs from previous runs: ~/.lightning_studio/logs/
#
# ── USAGE ────────────────────────────────────────────────────────────────────
#
#   Interactive development:
#     MODE=interactive bash on_start_cpu.sh
#
#   Test-only on a checkpoint:
#     MODE=test CHECKPOINT=/path/to/checkpoint.ckpt bash on_start_cpu.sh
#
#   Select dataset (default: isic2017):
#     DATASET=ph2      MODE=interactive bash on_start_cpu.sh
#     DATASET=isic2017 MODE=test CHECKPOINT=... bash on_start_cpu.sh
#
# ── INPUT VARIABLES ──────────────────────────────────────────────────────────
#
#   MODE              Required. One of: interactive | test
#   CHECKPOINT        Required when MODE=test. Path to .ckpt relative to repo root.
#
#   -- dataset --
#   DATASET           Dataset to use. One of: ph2 | isic2017. Default: isic2017
#                     ph2:      read from Lightning Storage (must be uploaded beforehand).
#                     isic2017: downloaded from Kaggle on first run, preprocessed once.
#   ISIC_OUT_DIR      Local download path for ISIC 2017.
#                     Default: /teamspace/lightning_storage/isic2017/ISIC2017DATA_256
#   KAGGLE_DATASET    Kaggle dataset slug for ISIC 2017.
#                     Default: johnchfr/isic-2017
#
#   -- repository --
#   REPO_URL          Git remote. Default: https://github.com/pkliui/SkiNet.git
#   BRANCH            Branch to check out. Default: train
#   HOST_REPO         Local clone path. Default: ~/repos/SkiNet
#   CONTAINER_REPO    Mount point inside the container. Default: /workplace/SkiNet
#
#   -- paths --
#   CONTAINER_MOUNT_PATH  Data mount point inside the container. Default: /mnt/data
#
#   -- misc --
#   PYTHON_BIN        Python binary. Default: python3
#
# ── NOTES ────────────────────────────────────────────────────────────────────
#
#   PH2:      data is read from Lightning Storage at /teamspace/lightning_storage/ph2/.
#             Ensure your data is uploaded there before running.
#
#   ISIC2017: data is downloaded from Kaggle to $ISIC_OUT_DIR on the first run.
#             Subsequent runs skip the download (directory non-empty) but always
#             re-run metadata CSV preprocessing. Requires the kaggle CLI and a
#             valid ~/.kaggle/kaggle.json credentials file.
#
# ─────────────────────────────────────────────────────────────────────────────

set -Eeuo pipefail

# ── Docker image ──────────────────────────────────────────────────────────────
IMAGE="pkliui/skinet:v9cpu"

# ── Mode of execution ─────────────────────────────────────────────────────────
MODE="${MODE:-interactive}"        # interactive | test
CHECKPOINT="${CHECKPOINT:-}"
CONFIG_FILE="${CONFIG_FILE:-main_config.yaml}"

# ── Repository ────────────────────────────────────────────────────────────────
REPO_URL="${REPO_URL:-https://github.com/pkliui/SkiNet.git}"
HOST_REPO="${HOST_REPO:-$HOME/repos/SkiNet}"
CONTAINER_REPO="${CONTAINER_REPO:-/workplace/SkiNet}"
BRANCH="${BRANCH:-train}"

# ── Python binary ─────────────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python3}"

# ── Data paths ────────────────────────────────────────────────────────────────
DATASET="${DATASET:-isic2017}"                            # ph2 | isic2017
CONTAINER_MOUNT_PATH="${CONTAINER_MOUNT_PATH:-/mnt/data}"

case "$DATASET" in
  ph2)
    LIGHTNING_MOUNT_PATH="/teamspace/lightning_storage/ph2/"
    ;;
  isic2017)
    ISIC_OUT_DIR="${ISIC_OUT_DIR:-/teamspace/lightning_storage/isic2017/ISIC2017DATA_256}"
    KAGGLE_DATASET="${KAGGLE_DATASET:-johnchfr/isic-2017}"
    LIGHTNING_MOUNT_PATH="$ISIC_OUT_DIR"
    ;;
  *)
    echo "ERROR: Unknown DATASET='$DATASET'. Valid values: ph2 | isic2017"
    exit 1
    ;;
esac

echo "==> Configuration:"
echo "    MODE=$MODE  CHECKPOINT='$CHECKPOINT'  CONFIG_FILE=$CONFIG_FILE"
echo "    REPO_URL=$REPO_URL  BRANCH=$BRANCH"
echo "    HOST_REPO=$HOST_REPO  CONTAINER_REPO=$CONTAINER_REPO"
echo "    DATASET=$DATASET  LIGHTNING_MOUNT_PATH=$LIGHTNING_MOUNT_PATH  CONTAINER_MOUNT_PATH=$CONTAINER_MOUNT_PATH"
echo "    PYTHON_BIN=$PYTHON_BIN"

for cmd in docker git; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: $cmd is not installed"
    exit 1
  fi
done

mkdir -p "$(dirname "$HOST_REPO")"

# Update repo if it exists, otherwise clone it
if [[ -d "$HOST_REPO/.git" ]]; then
  echo "==> Repo already exists, updating it"
  git -C "$HOST_REPO" fetch origin
  git -C "$HOST_REPO" checkout "$BRANCH"
  git -C "$HOST_REPO" pull --ff-only origin "$BRANCH"
else
  if [[ -d "$HOST_REPO" ]] && [[ -n "$(ls -A "$HOST_REPO" 2>/dev/null)" ]]; then
    echo "ERROR: $HOST_REPO exists but is not a git repo and is not empty"
    exit 1
  fi
  echo "==> Cloning repo into $HOST_REPO"
  rm -rf "$HOST_REPO"
  git clone "$REPO_URL" "$HOST_REPO"
  git -C "$HOST_REPO" checkout "$BRANCH"
fi

echo "==> Pulling Docker image $IMAGE"
docker pull "$IMAGE"

# ── ISIC 2017: download + preprocess ─────────────────────────────────────────
# Only runs when DATASET=isic2017. Download is skipped when data is already
# present; preprocessing always runs (it is idempotent and fast).

if [[ "$DATASET" == "isic2017" ]]; then
  if ! command -v kaggle >/dev/null 2>&1; then
    echo "==> kaggle CLI not found — installing..."
    pip install --quiet kaggle
  fi

  mkdir -p "$ISIC_OUT_DIR"

  if [[ -z "$(ls -A "$ISIC_OUT_DIR" 2>/dev/null)" ]]; then
    echo "==> ISIC 2017: directory empty — downloading dataset to $ISIC_OUT_DIR"
    kaggle datasets download -d "$KAGGLE_DATASET" -p "$ISIC_OUT_DIR" --unzip
  else
    echo "==> ISIC 2017: data already present in $ISIC_OUT_DIR — skipping download."
  fi

  echo "==> ISIC 2017: running metadata CSV preprocessing..."
  docker run --rm \
      --user "$(id -u):$(id -g)" \
      -e LOGNAME="${LOGNAME:-user}" \
      -e USER="${USER:-user}" \
      -e HOME="$CONTAINER_REPO" \
      --mount "type=bind,src=$HOST_REPO,dst=$CONTAINER_REPO" \
      --mount "type=bind,src=$ISIC_OUT_DIR,dst=$CONTAINER_MOUNT_PATH" \
      -w "$CONTAINER_REPO" \
      "$IMAGE" \
      "$PYTHON_BIN" -m SkiNet.ML.datasets.preprocessing.metadata_csv_factory \
      --dataset-key-str ISIC2017 \
      --local-data-root "$CONTAINER_MOUNT_PATH"

  echo "==> ISIC 2017: setup complete."
fi

LIGHTNING_ENV_FILE="$(mktemp)"
env | grep '^LIGHTNING_' > "$LIGHTNING_ENV_FILE" || true


echo "==> Preparing container..."

# Kill any existing process holding MLflow's port
if lsof -ti:5000 &>/dev/null; then
  echo "==> Port 5000 in use — killing existing process..."
  kill -9 "$(lsof -ti:5000)" 2>/dev/null || true
  sleep 1
fi

if [[ "$MODE" == "test" ]]; then
  if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: MODE=test requires CHECKPOINT to be set"
    exit 1
  fi
  # Map the host checkpoint path into the container under the repo mount
  CONTAINER_CHECKPOINT="$CONTAINER_REPO/$(realpath --relative-to="$HOST_REPO" "$CHECKPOINT")"
  echo "==> Running test-only with checkpoint: $CONTAINER_CHECKPOINT"
  docker run --rm \
    -p 5000:5000 \
    --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e HOME="$CONTAINER_REPO" \
    --env-file "$LIGHTNING_ENV_FILE" \
    -v "$HOME/.lightning:$CONTAINER_REPO/.lightning:ro" \
    --mount "type=bind,src=$HOST_REPO,dst=$CONTAINER_REPO" \
    --mount "type=bind,src=$LIGHTNING_MOUNT_PATH,dst=$CONTAINER_MOUNT_PATH" \
    -w "$CONTAINER_REPO" \
    "$IMAGE" \
    bash -c "
    set -e
    ./start_mlflow.sh &
    sleep 3
    $PYTHON_BIN -u main_run.py --config $CONFIG_FILE --test-only --checkpoint '$CONTAINER_CHECKPOINT'"
elif [[ "$MODE" == "interactive" ]]; then
  docker run --rm -it \
    -p 5000:5000 \
    --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e HOME="$CONTAINER_REPO" \
    --env-file "$LIGHTNING_ENV_FILE" \
    -v "$HOME/.lightning:$CONTAINER_REPO/.lightning:ro" \
    --mount "type=bind,src=$HOST_REPO,dst=$CONTAINER_REPO" \
    --mount "type=bind,src=$LIGHTNING_MOUNT_PATH,dst=$CONTAINER_MOUNT_PATH" \
    -w "$CONTAINER_REPO" \
    "$IMAGE" \
    bash -c "
    set -e
    ./start_mlflow.sh &
    sleep 3
    exec bash"
else
  echo "ERROR: Unknown MODE='$MODE'. Valid values: interactive, test"
  exit 1
fi
rm -f "$LIGHTNING_ENV_FILE"
echo "==> Done. Docker is running. MLFlow is available at http://localhost:5000"
echo "==> Attach Shell to the running container in VSC Containers tab to develop inside the container"
