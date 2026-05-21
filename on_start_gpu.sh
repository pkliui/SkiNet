#!/usr/bin/env bash
#
# on_start_gpu.sh — SkiNet training launcher for Lightning Studio GPU sessions.
#
# Clones/updates the SkiNet repo, pulls the Docker image, launches a container
# with GPU access and Lightning Storage mounted, starts MLflow, and runs the
# requested training job. Optionally releases the GPU when done.
#
# Logs from previous runs: ~/.lightning_studio/logs/
#
# ── USAGE ────────────────────────────────────────────────────────────────────
#
#   Single training run:
#     MODE=train bash on_start_gpu.sh
#
#   Multi-seed run with YAML-defined encoder/merge modes:
#     MODE=seeds SEEDS="1 2 3 4 5" bash on_start_gpu.sh
#
#   Multi-seed run with explicit encoder/merge modes (overrides YAML config):
#     RUN_TRAINING=true DATASET=isic2017 ENCODER_MODES="classical" MERGE_MODES="classical local_refinement he2 attention_gate" MODE=seeds SEEDS="100 101 102 103 104" bash on_start_gpu.sh
#
#   Optuna sweep with non-default metric:
#     MODE=sweep SWEEP_MONITOR=val_best_dice_at_threshold SWEEP_DIRECTION=maximize bash on_start_gpu.sh
#
#   Dry run — build command but skip container launch:
#     RUN_TRAINING=false MODE=train bash on_start_gpu.sh
#
#   Keep GPU after training (default: release):
#     RELEASE_GPU=false MODE=train bash on_start_gpu.sh
#
#   Select dataset (default: ph2):
#     DATASET=ph2      MODE=seeds SEEDS="1 2 3" bash on_start_gpu.sh
#     DATASET=isic2017 MODE=seeds SEEDS="1 2 3" bash on_start_gpu.sh
#
# ── INPUT VARIABLES ──────────────────────────────────────────────────────────
#
#   MODE              Required. One of: train | seeds | sweep
#   RUN_TRAINING      Launch the container. Default: true
#   RELEASE_GPU       Switch to CPU after training. Default: true
#
#   CONFIG_FILE       YAML config filename (resolved inside the container workdir).
#                     Default: main_config.yaml
#
#   -- seeds mode --
#   SEEDS             Required. Space-separated seed values (run_seeds.py has no default).
#   ENCODER_MODES     Space-separated encoder modes. Omit to use YAML config value.
#   MERGE_MODES       Space-separated merge modes.   Omit to use YAML config value.
#
#   -- sweep mode --
#   SWEEP_MONITOR     Metric to optimise. Default: val_best_dice_at_threshold
#   SWEEP_DIRECTION   maximize | minimize.  Default: maximize
#
#   -- dataset --
#   DATASET           Dataset to use. One of: ph2 | isic2017. Default: ph2
#                     ph2:      read from Lightning Storage (must be uploaded beforehand).
#                     isic2017: downloaded from Kaggle on first run, preprocessed once.
#   ISIC_OUT_DIR      Local download path for ISIC 2017.
#                     Default: /teamspace/studios/this_studio/isic2017
#   KAGGLE_DATASET    Kaggle dataset slug for ISIC 2017.
#                     Default: johnchfr/isic-2017
#
#   -- repository --
#   REPO_URL          Git remote. Default: https://github.com/pkliui/SkiNet.git
#   BRANCH            Branch to check out. Default: train
#   HOST_REPO         Local clone path.   Default: ~/repos/SkiNet
#   CONTAINER_REPO    Mount point inside the container. Default: /workplace/SkiNet
#
#   -- paths --
#   CONTAINER_MOUNT_PATH  Data mount point inside the container. Default: /mnt/data
#
#   -- misc --
#   PYTHON_BIN        Python binary. Default: python
#
# ── NOTES ────────────────────────────────────────────────────────────────────
#
#   PH2:      data is read from Lightning Storage at /teamspace/lightning_storage/ph2/.
#             Ensure your data is uploaded there before running (path configured in
#             repos/SkiNet/SkiNet/Azure/azure_settings.yaml under PATH_ON_DATASTORE).
#
#   ISIC2017: data is downloaded from Kaggle to $ISIC_OUT_DIR on the first run.
#             Subsequent runs skip the download (directory non-empty) but always
#             re-run metadata CSV preprocessing. Requires the kaggle CLI and a
#             valid ~/.kaggle/kaggle.json credentials file.
#
# ─────────────────────────────────────────────────────────────────────────────

set -Eeuo pipefail

# ── Docker image ──────────────────────────────────────────────────────────────
IMAGE="pkliui/skinet:v9gpu"

# ── Training control ──────────────────────────────────────────────────────────
RUN_TRAINING="${RUN_TRAINING:-}"
RELEASE_GPU="${RELEASE_GPU:-false}"
MODE="${MODE:-}"

# ── Config file ───────────────────────────────────────────────────────────────
CONFIG_FILE="${CONFIG_FILE:-main_config.yaml}"

# ── Mode-specific arguments ───────────────────────────────────────────────────
SEEDS="${SEEDS:-}"                 # seeds mode: required — run_seeds.py has no default
ENCODER_MODES="${ENCODER_MODES:-}" # seeds mode: overrides YAML encoder_mode if set
MERGE_MODES="${MERGE_MODES:-}"     # seeds mode: overrides YAML merge_mode if set

SWEEP_MONITOR="${SWEEP_MONITOR:-val_best_dice_at_threshold}" # sweep mode: metric to optimise perf/samples_per_sec, val_best_dice_at_threshold
SWEEP_DIRECTION="${SWEEP_DIRECTION:-maximize}"               # sweep mode: maximize | minimize

# ── Repository ────────────────────────────────────────────────────────────────
REPO_URL="${REPO_URL:-https://github.com/pkliui/SkiNet.git}"
HOST_REPO="${HOST_REPO:-$HOME/repos/SkiNet}"
CONTAINER_REPO="${CONTAINER_REPO:-/workplace/SkiNet}"
BRANCH="${BRANCH:-train}"

# ── Data paths ────────────────────────────────────────────────────────────────
DATASET="${DATASET:-isic2017}"                  # ph2 | isic2017
CONTAINER_MOUNT_PATH="${CONTAINER_MOUNT_PATH:-/mnt/data}" # "/kaggle/working/isic2017_data_256/"

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

# ── Python binary ─────────────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"

# ── Internal state ────────────────────────────────────────────────────────────
LIGHTNING_ENV_FILE=""
CONTAINER_NAME=""
_GPU_RELEASED=false

# ── Config summary ────────────────────────────────────────────────────────────
echo "==> Configuration:"
echo "    MODE=$MODE  RUN_TRAINING=$RUN_TRAINING  RELEASE_GPU=$RELEASE_GPU"
echo "    CONFIG_FILE=$CONFIG_FILE"
echo "    SEEDS='$SEEDS'  ENCODER_MODES='$ENCODER_MODES'  MERGE_MODES='$MERGE_MODES'"
echo "    SWEEP_MONITOR=$SWEEP_MONITOR  SWEEP_DIRECTION=$SWEEP_DIRECTION"
echo "    REPO_URL=$REPO_URL  BRANCH=$BRANCH"
echo "    HOST_REPO=$HOST_REPO  CONTAINER_REPO=$CONTAINER_REPO"
echo "    DATASET=$DATASET  LIGHTNING_MOUNT_PATH=$LIGHTNING_MOUNT_PATH  CONTAINER_MOUNT_PATH=$CONTAINER_MOUNT_PATH"
echo "    PYTHON_BIN=$PYTHON_BIN"

# ── Helpers ───────────────────────────────────────────────────────────────────

release_gpu() {
  [[ "$_GPU_RELEASED" == "true" ]] && return
  _GPU_RELEASED=true
  if [[ "$RELEASE_GPU" != "true" ]]; then
    echo "==> RELEASE_GPU=false — skipping GPU release."
    return
  fi
  echo "==> Releasing GPU..."
  python3 -c "
from lightning_sdk import Studio, Machine
studio = Studio()
try:
    studio.switch_machine(Machine.CPU)
    print('GPU released — switched to CPU successfully.')
except Exception as e:
    print(f'switch_machine(CPU) failed: {e}')
    print('Please switch to CPU manually in the Lightning UI.')
"
}

cleanup() {
  local sig="${1:-EXIT}"
  [[ -n "$LIGHTNING_ENV_FILE" ]] && rm -f "$LIGHTNING_ENV_FILE"
  if [[ "$sig" == "INT" || "$sig" == "TERM" ]]; then
    echo ""
    echo "==> Interrupted ($sig). Stopping container if running..."
    if [[ -n "$CONTAINER_NAME" ]]; then
      docker stop "$CONTAINER_NAME" 2>/dev/null || true
      docker rm   "$CONTAINER_NAME" 2>/dev/null || true
      echo "==> Container '$CONTAINER_NAME' removed."
    fi
    echo "==> Waiting for GPU to detach..."
    sleep 10
  fi
  release_gpu
}

trap 'cleanup EXIT' EXIT
trap 'cleanup INT;  exit 130' INT
trap 'cleanup TERM; exit 143' TERM

# ── Pre-flight checks ─────────────────────────────────────────────────────────

for cmd in docker git; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: $cmd is not installed"
    exit 1
  fi
done

# ── Clone or update repo ──────────────────────────────────────────────────────

mkdir -p "$(dirname "$HOST_REPO")"

if [[ -d "$HOST_REPO/.git" ]]; then
  echo "==> Repo exists — fetching and updating branch '$BRANCH'"
  git -C "$HOST_REPO" fetch origin
  git -C "$HOST_REPO" checkout "$BRANCH"
  git -C "$HOST_REPO" pull --ff-only origin "$BRANCH"
else
  if [[ -d "$HOST_REPO" ]] && [[ -n "$(ls -A "$HOST_REPO" 2>/dev/null)" ]]; then
    echo "ERROR: $HOST_REPO exists but is not a git repo and is not empty"
    exit 1
  fi
  echo "==> Cloning $REPO_URL into $HOST_REPO (branch: $BRANCH)"
  rm -rf "$HOST_REPO"
  git clone "$REPO_URL" "$HOST_REPO"
  git -C "$HOST_REPO" checkout "$BRANCH"
fi

# ── Pull Docker image ─────────────────────────────────────────────────────────

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


# ── Collect Lightning env vars ────────────────────────────────────────────────

LIGHTNING_ENV_FILE="$(mktemp)"
env | grep '^LIGHTNING_' > "$LIGHTNING_ENV_FILE" || true

# ── Build Python command ──────────────────────────────────────────────────────

echo "==> Building Python command for MODE='$MODE'"

if [[ "$MODE" == "sweep" ]]; then
  PYTHON_CMD="$PYTHON_BIN -u optuna_sweep.py --config $CONFIG_FILE --monitor $SWEEP_MONITOR --direction $SWEEP_DIRECTION"

elif [[ "$MODE" == "seeds" ]]; then
  if [[ -z "$SEEDS" ]]; then
    echo "ERROR: MODE=seeds requires SEEDS to be set (run_seeds.py has no default)"
    exit 1
  fi
  PYTHON_CMD="$PYTHON_BIN -u run_seeds.py --config $CONFIG_FILE --seeds $SEEDS"
  [[ -n "$ENCODER_MODES" ]] && PYTHON_CMD="$PYTHON_CMD --encoder-modes $ENCODER_MODES"
  [[ -n "$MERGE_MODES" ]]   && PYTHON_CMD="$PYTHON_CMD --merge-modes $MERGE_MODES"

elif [[ "$MODE" == "train" ]]; then
  PYTHON_CMD="$PYTHON_BIN -u main_run.py --config $CONFIG_FILE"

else
  echo "ERROR: Unknown MODE='$MODE'. Valid values: train | seeds | sweep"
  exit 1
fi

echo "==> Command: $PYTHON_CMD"

# ── Launch container ──────────────────────────────────────────────────────────

if [[ "$RUN_TRAINING" != "true" ]]; then
  echo "==> RUN_TRAINING=false — skipping container launch."
  echo "==> To launch manually: docker run ... $PYTHON_CMD"
  exit 0
fi

echo "==> Preparing container..."

# Kill any existing process holding MLflow's port
if lsof -ti:5000 &>/dev/null; then
  echo "==> Port 5000 in use — killing existing process..."
  kill -9 "$(lsof -ti:5000)" 2>/dev/null || true
  sleep 1
fi

mkdir -p "$HOST_REPO/mlruns"

CONTAINER_NAME="skinet-$(date +%s)"
echo "==> Starting container '$CONTAINER_NAME' (MLflow → host:5000, training non-interactive)"

docker run --name "$CONTAINER_NAME" \
  -p 5000:5000 \
  --gpus all \
  --ipc=host \
  --user "$(id -u):$(id -g)" \
  -e HOME="$CONTAINER_REPO" \
  -e LOGNAME="${LOGNAME:-user}" \
  -e USER="${USER:-user}" \
  -e PYTHONUNBUFFERED=1 \
  -e MLFLOW_HOST_ARTIFACT_PATH="$HOST_REPO/mlruns" \
  --env-file "$LIGHTNING_ENV_FILE" \
  -v "$HOME/.lightning:$CONTAINER_REPO/.lightning:ro" \
  --mount "type=bind,src=$HOST_REPO,dst=$CONTAINER_REPO" \
  --mount "type=bind,src=$HOST_REPO/mlruns,dst=$HOST_REPO/mlruns" \
  --mount "type=bind,src=$LIGHTNING_MOUNT_PATH,dst=$CONTAINER_MOUNT_PATH" \
  -w "$CONTAINER_REPO" \
  "$IMAGE" \
  bash -c "
    set -e
    ./start_mlflow.sh
    exec $PYTHON_CMD
  " && DOCKER_EXIT=0 || DOCKER_EXIT=$?

echo "==> Docker exited with code $DOCKER_EXIT"

if [[ $DOCKER_EXIT -eq 0 ]]; then
  echo "==> Training succeeded — removing container..."
  docker rm "$CONTAINER_NAME"
  CONTAINER_NAME=""
  echo "==> Waiting for GPU to detach..."
  sleep 10
  release_gpu
  echo "==> Done."
else
  echo "==> Container '$CONTAINER_NAME' failed (exit $DOCKER_EXIT) — keeping for debugging."
  echo "    Re-enter:  docker start $CONTAINER_NAME && docker exec -it $CONTAINER_NAME bash"
  echo "    Clean up:  docker rm $CONTAINER_NAME"
  echo "==> Waiting for GPU to detach..."
  sleep 10
  release_gpu
fi
