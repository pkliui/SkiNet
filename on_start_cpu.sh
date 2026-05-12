#!/bin/bash

# This script runs every time your Studio starts, from your home directory.

# Logs from previous runs can be found in ~/.lightning_studio/logs/

# List files under fast_load that need to load quickly on start (e.g. model checkpoints).
#
# ! fast_load
# <your file here>

# Add your startup commands below.
#
# Example: streamlit run my_app.py
# Example: gradio my_app.py

#!/usr/bin/env bash

# This script is intended to be run on Lightning Studio machine with CPU. It will:
# 1. Clone the SkiNet repo (or update it if it already exists)
# 2. Pull the specified Docker image
# 3. Run a container from the image, mounting the repo and the Lightning Storage folder into the container
# 4. Start MLFlow server in the background in this container (start_mlflow.sh is expected in the same as this script)
# 5.Depending on command MODE (see example), container is either available for interactive development
# or will do a test-only run.

##########################################
#  --------------- USAGE ------------------

# Run container with CPU in interactive development mode
# MODE=interactive bash on_start_cpu.sh

# Run test-only on a checkpoint
# MODE=test CHECKPOINT=/path/to/checkpoint.ckpt bash on_start_cpu.sh

########################################################################################################

# NOTE: It is assumed that you uploaded your data into the Lightning Storage and that it is in folders
# specified under PATH_ON_DATASTORE in repos/SkiNet/SkiNet/Azure/azure_settings.yaml

########################################################################################################

set -Eeuo pipefail

# Set default values for environment variables if they are not already set

# Mode of execution
MODE="${MODE:-interactive}" # "interactive" = set up container, "test" = test-only run
CHECKPOINT="${CHECKPOINT:-}"


# Image name on Docker Hub
IMAGE="pkliui/skinet:v9cpu"

# Determine a safe default for the home directory
DEFAULT_HOME="$HOME"

# Set repository variables
REPO_URL="${REPO_URL:-https://github.com/pkliui/SkiNet.git}"
HOST_REPO="${HOST_REPO:-$DEFAULT_HOME/repos/SkiNet}"
CONTAINER_REPO="${CONTAINER_REPO:-/workplace/SkiNet}"
BRANCH="${BRANCH:-train}"

# Set Python binary
PYTHON_BIN="${PYTHON_BIN:-python3}"


# Data mount path on Lightning Storage
#LIGHTNING_MOUNT_PATH="/teamspace/lightning_storage/ph2_002-032/"
#LIGHTNING_MOUNT_PATH="/teamspace/lightning_storage/ph2/"
#LIGHTNING_MOUNT_PATH="/teamspace/lightning_storage/isic2017/"
LIGHTNING_MOUNT_PATH="/teamspace/studios/this_studio/isic2017/"
# Data mount path inside the container
CONTAINER_MOUNT_PATH="${CONTAINER_MOUNT_PATH:-/mnt/data}"

echo "Running with the following configuration:"
echo "DEFAULT_HOME=$DEFAULT_HOME"
echo "REPO_URL=$REPO_URL"
echo "HOST_REPO=$HOST_REPO"
echo "CONTAINER_REPO=$CONTAINER_REPO"
echo "BRANCH=$BRANCH"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "LIGHTNING_MOUNT_PATH=$LIGHTNING_MOUNT_PATH"
echo "CONTAINER_MOUNT_PATH=$CONTAINER_MOUNT_PATH"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not installed"
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is not installed"
  exit 1
fi

# Clone or update the repo on the host
HOST_REPO_PARENT="$(dirname "$HOST_REPO")"
mkdir -p "$HOST_REPO_PARENT"

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

echo "Setting up Lightning"
LIGHTNING_ENV_FILE="$(mktemp)"
env | grep '^LIGHTNING_' > "$LIGHTNING_ENV_FILE" || true


echo "==> Preparing to run a fresh container"

# Kill any existing MLflow process on port 5000
if lsof -ti:5000 &>/dev/null; then
  echo "Port 5000 in use — killing existing process..."
  kill -9 $(lsof -ti:5000) 2>/dev/null || true
  sleep 1
fi

echo "==> Setting MLflow port mapping host:container to 5000:5000"
echo "==> RUNNING container and MLflow server in background"

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
    python3 -u main_run.py --config main_config.yaml --test-only --checkpoint '$CONTAINER_CHECKPOINT'"
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
