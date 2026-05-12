#!/usr/bin/env bash
#
# on_start_kaggle.sh — SkiNet training launcher for Kaggle GPU sessions.
#
# Clones/updates the SkiNet repo, starts MLflow, and runs the requested
# training job directly in the Kaggle Python environment (no Docker).
#
# ── USAGE ────────────────────────────────────────────────────────────────────
#
#   Single training run:
#     MODE=train bash on_start_kaggle.sh
#
#   Multi-seed run with YAML-defined encoder/merge modes:
#     MODE=seeds SEEDS="1 2 3 4 5" bash on_start_kaggle.sh
#
#   Multi-seed run with explicit encoder/merge modes (overrides YAML config):
#     ENCODER_MODES="local_refinement he2 se" MERGE_MODES="local_refinement he2 attention_gate" MODE=seeds SEEDS="1 2 3 4 5" bash on_start_kaggle.sh
#
#   Optuna sweep with non-default metric:
#     MODE=sweep SWEEP_MONITOR=val_best_dice_at_threshold SWEEP_DIRECTION=maximize bash on_start_kaggle.sh
#
#   Dry run — print command but skip execution:
#     RUN_TRAINING=false MODE=train bash on_start_kaggle.sh
#
#   Select dataset (default: ph2):
#     DATASET=ph2      MODE=seeds SEEDS="1 2 3" bash on_start_kaggle.sh
#     DATASET=isic2017 MODE=seeds SEEDS="1 2 3" bash on_start_kaggle.sh
#
# ── INPUT VARIABLES ──────────────────────────────────────────────────────────
#
#   MODE              Required. One of: train | seeds | sweep
#   RUN_TRAINING      Execute the training command. Default: true
#
#   CONFIG_FILE       YAML config filename (resolved inside the repo workdir).
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
#                     ph2:      read from Kaggle input path (DATA_DIR).
#                     isic2017: read from Kaggle input path (DATA_DIR);
#                               preprocessing always runs (idempotent and fast).
#   DATA_DIR          Path to the dataset on disk.
#                     ph2 default:      /kaggle/input/ph2
#                     isic2017 default: /kaggle/input/isic-2017
#
#   -- repository --
#   REPO_URL          Git remote. Default: https://github.com/pkliui/SkiNet.git
#   GITHUB_TOKEN      Personal Access Token for private repos. Add it as a Kaggle
#                     secret (Settings → Secrets) and expose it to the notebook.
#                     The token is injected into the clone URL and then scrubbed
#                     from .git/config so it is not persisted on disk.
#   BRANCH            Branch to check out. Default: train
#   HOST_REPO         Local clone path. Default: /kaggle/working/SkiNet
#
#   -- paths --
#   CONTAINER_MOUNT_PATH  Data path expected by training scripts. Default: /mnt/data
#                         Training scripts read data from this path; a symlink is
#                         created from CONTAINER_MOUNT_PATH → DATA_DIR if they differ.
#
#   -- misc --
#   PYTHON_BIN        Python binary. Default: python
#
# ── NOTES ────────────────────────────────────────────────────────────────────
#
#   PH2 / ISIC 2017:  add the Kaggle dataset to your notebook as an input so it
#                     appears under /kaggle/input/. Set DATA_DIR if the mount
#                     path differs from the default.
#
#   MLflow:           tracking server starts on localhost:5000. Access it via
#                     the Kaggle session's port-forwarding or inspect
#                     /kaggle/working/SkiNet/mlruns directly after the run.
#
# ─────────────────────────────────────────────────────────────────────────────

set -Eeuo pipefail

# ── Training control ──────────────────────────────────────────────────────────
RUN_TRAINING="${RUN_TRAINING:-true}"
MODE="${MODE:-train}"

# ── Config file ───────────────────────────────────────────────────────────────
CONFIG_FILE="${CONFIG_FILE:-main_config.yaml}"

# ── Mode-specific arguments ───────────────────────────────────────────────────
SEEDS="${SEEDS:-}"
ENCODER_MODES="${ENCODER_MODES:-}"
MERGE_MODES="${MERGE_MODES:-}"

SWEEP_MONITOR="${SWEEP_MONITOR:-perf/samples_per_sec}"
SWEEP_DIRECTION="${SWEEP_DIRECTION:-maximize}"

# ── Repository ────────────────────────────────────────────────────────────────
REPO_URL="${REPO_URL:-https://github.com/pkliui/SkiNet.git}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"   # set as a Kaggle secret; injected into URL at clone time
HOST_REPO="${HOST_REPO:-/kaggle/working/SkiNet}"
BRANCH="${BRANCH:-train}"

# ── Data paths ────────────────────────────────────────────────────────────────
DATASET="${DATASET:-isic2017}"
CONTAINER_MOUNT_PATH="${CONTAINER_MOUNT_PATH:-/mnt/data}"

case "$DATASET" in
  ph2)
    DATA_DIR="${DATA_DIR:-/kaggle/input/ph2}"
    ;;
  isic2017)
    DATA_DIR="${DATA_DIR:-/kaggle/input/datasets/johnchfr/isic-2017}"
    ;;
  *)
    echo "ERROR: Unknown DATASET='$DATASET'. Valid values: ph2 | isic2017"
    exit 1
    ;;
esac

# ── Python binary ─────────────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"

# ── Config summary ────────────────────────────────────────────────────────────
echo "==> Configuration:"
echo "    MODE=$MODE  RUN_TRAINING=$RUN_TRAINING"
echo "    CONFIG_FILE=$CONFIG_FILE"
echo "    SEEDS='$SEEDS'  ENCODER_MODES='$ENCODER_MODES'  MERGE_MODES='$MERGE_MODES'"
echo "    SWEEP_MONITOR=$SWEEP_MONITOR  SWEEP_DIRECTION=$SWEEP_DIRECTION"
echo "    REPO_URL=$REPO_URL  BRANCH=$BRANCH"
echo "    HOST_REPO=$HOST_REPO"
echo "    DATASET=$DATASET  DATA_DIR=$DATA_DIR  CONTAINER_MOUNT_PATH=$CONTAINER_MOUNT_PATH"
echo "    PYTHON_BIN=$PYTHON_BIN"

# ── Helpers ───────────────────────────────────────────────────────────────────

cleanup() {
  local sig="${1:-EXIT}"
  if [[ "$sig" == "INT" || "$sig" == "TERM" ]]; then
    echo ""
    echo "==> Interrupted ($sig) — stopping any background MLflow process..."
    # Kill mlflow if it was started in the background
    pkill -f "mlflow server" 2>/dev/null || true
  fi
}

trap 'cleanup EXIT' EXIT
trap 'cleanup INT;  exit 130' INT
trap 'cleanup TERM; exit 143' TERM

# ── Pre-flight checks ─────────────────────────────────────────────────────────

for cmd in git "$PYTHON_BIN"; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: $cmd is not installed"
    exit 1
  fi
done

if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: DATA_DIR='$DATA_DIR' does not exist."
  echo "       Add the dataset as a Kaggle input or set DATA_DIR to the correct path."
  exit 1
fi

# ── Clone or update repo ──────────────────────────────────────────────────────

mkdir -p "$(dirname "$HOST_REPO")"

# Inject token into URL for private repos (token never printed — URL not echoed)
_AUTH_REPO_URL="$REPO_URL"
if [[ -n "$GITHUB_TOKEN" ]]; then
  _AUTH_REPO_URL="${REPO_URL/https:\/\//https:\/\/${GITHUB_TOKEN}@}"
fi

if [[ -d "$HOST_REPO/.git" ]]; then
  echo "==> Repo exists — fetching and updating branch '$BRANCH'"
  git -C "$HOST_REPO" remote set-url origin "$_AUTH_REPO_URL"
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
  git clone "$_AUTH_REPO_URL" "$HOST_REPO"
  # Replace token-embedded URL with the clean URL in .git/config
  git -C "$HOST_REPO" remote set-url origin "$REPO_URL"
  git -C "$HOST_REPO" checkout "$BRANCH"
fi

# ── Symlink data path ─────────────────────────────────────────────────────────
# Training scripts expect data at CONTAINER_MOUNT_PATH. If DATA_DIR differs,
# create a symlink so the scripts find their data without any code changes.

if [[ "$CONTAINER_MOUNT_PATH" != "$DATA_DIR" ]]; then
  if [[ -L "$CONTAINER_MOUNT_PATH" ]]; then
    echo "==> Updating symlink $CONTAINER_MOUNT_PATH → $DATA_DIR"
    ln -sfn "$DATA_DIR" "$CONTAINER_MOUNT_PATH"
  elif [[ -e "$CONTAINER_MOUNT_PATH" ]]; then
    echo "ERROR: $CONTAINER_MOUNT_PATH already exists and is not a symlink — cannot create data symlink."
    exit 1
  else
    echo "==> Creating symlink $CONTAINER_MOUNT_PATH → $DATA_DIR"
    mkdir -p "$(dirname "$CONTAINER_MOUNT_PATH")"
    ln -s "$DATA_DIR" "$CONTAINER_MOUNT_PATH"
  fi
fi

# ── ISIC 2017: preprocessing ──────────────────────────────────────────────────
# Data is already present via Kaggle input — only preprocessing is needed.
# The preprocessor writes isic2017_metadata.csv into data_root, so data_root
# must be writable. /kaggle/input is read-only, so we copy the dataset to a
# writable staging dir and re-point the symlink before preprocessing.

if [[ "$DATASET" == "isic2017" ]]; then
  ISIC2017_WRITABLE_DIR="/kaggle/working/isic2017_data"
  if [[ ! -d "$ISIC2017_WRITABLE_DIR" ]]; then
    echo "==> ISIC 2017: copying dataset to writable path $ISIC2017_WRITABLE_DIR ..."
    cp -r "$DATA_DIR/." "$ISIC2017_WRITABLE_DIR"
    echo "==> ISIC 2017: copy complete."
  else
    echo "==> ISIC 2017: writable copy already exists at $ISIC2017_WRITABLE_DIR"
  fi
  # Re-point the symlink (or update DATA_DIR) so training also reads from writable path
  ln -sfn "$ISIC2017_WRITABLE_DIR" "$CONTAINER_MOUNT_PATH"
  echo "==> ISIC 2017: running metadata CSV preprocessing..."
  cd "$HOST_REPO"
  "$PYTHON_BIN" -m SkiNet.ML.datasets.preprocessing.metadata_csv_factory \
    --dataset-key-str ISIC2017 \
    --local-data-root "$ISIC2017_WRITABLE_DIR"
  echo "==> ISIC 2017: preprocessing complete."
fi

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

# ── Dry run ───────────────────────────────────────────────────────────────────

if [[ "$RUN_TRAINING" != "true" ]]; then
  echo "==> RUN_TRAINING=false — skipping execution."
  echo "==> To run manually from $HOST_REPO: $PYTHON_CMD"
  exit 0
fi

# ── Start MLflow ──────────────────────────────────────────────────────────────

mkdir -p "$HOST_REPO/mlruns"
cd "$HOST_REPO"

# Kill any existing process holding MLflow's port
if lsof -ti:5000 &>/dev/null; then
  echo "==> Port 5000 in use — killing existing process..."
  kill -9 "$(lsof -ti:5000)" 2>/dev/null || true
  sleep 1
fi

echo "==> Starting MLflow tracking server (localhost:5000)..."
bash ./start_mlflow.sh &

# ── Run training ──────────────────────────────────────────────────────────────

echo "==> Launching training: $PYTHON_CMD"
TRAIN_EXIT=0
eval "$PYTHON_CMD" || TRAIN_EXIT=$?

echo "==> Training exited with code $TRAIN_EXIT"

if [[ $TRAIN_EXIT -eq 0 ]]; then
  echo "==> Training succeeded."
else
  echo "==> Training failed (exit $TRAIN_EXIT)."
  echo "    Inspect mlruns at $HOST_REPO/mlruns"
  exit $TRAIN_EXIT
fi
