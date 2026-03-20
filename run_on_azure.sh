#!/usr/bin/env bash

# This script is intended to be run on an Azure VM. It will:
# 1. Clone the SkiNet repo (or update it if it already exists)
# 2. Install blobfuse2 if it is not already installed
# 3. Mount the Azure Blob Storage using blobfuse2
# 4. Pull the specified Docker image
# 5. Run a container from the image, mounting the repo and the Azure Blob Storage into the container

set -Eeuo pipefail

IMAGE="${1:?Usage: $0 <docker-image> [script args...]}"
shift || true

# Set default values for environment variables if they are not already set

# Determine a safe default for the home directory
DEFAULT_HOME="/home/azureuser"

# Set repository variables
REPO_URL="${REPO_URL:-https://github.com/pkliui/SkiNet.git}"
HOST_REPO="${HOST_REPO:-$DEFAULT_HOME/repos/SkiNet}"
CONTAINER_REPO="${CONTAINER_REPO:-/workplace/SkiNet}"
BRANCH="${BRANCH:-dev}"

# Set Python binary
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Azure managed identity ID
AZURE_MANAGED_IDENTITY_CLIENT_ID="${AZURE_MANAGED_IDENTITY_CLIENT_ID:-}"
# Data mount path on Azure VM - must coincide with the mount path used in mount_data.py
AZURE_MOUNT_PATH="${AZURE_MOUNT_PATH:-$DEFAULT_HOME/mnt/azure_blob_data}"
# Data mount path inside the container
CONTAINER_AZURE_MOUNT_PATH="${CONTAINER_AZURE_MOUNT_PATH:-/mnt/data}"

echo "Running with the following configuration:"
echo "DEFAULT_HOME=$DEFAULT_HOME"
echo "REPO_URL=$REPO_URL"
echo "HOST_REPO=$HOST_REPO"
echo "CONTAINER_REPO=$CONTAINER_REPO"
echo "BRANCH=$BRANCH"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "AZURE_MANAGED_IDENTITY_CLIENT_ID=$AZURE_MANAGED_IDENTITY_CLIENT_ID"
echo "AZURE_MOUNT_PATH=$AZURE_MOUNT_PATH"
echo "CONTAINER_AZURE_MOUNT_PATH=$CONTAINER_AZURE_MOUNT_PATH"

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

# Install blobfuse2
if ! command -v blobfuse2 >/dev/null 2>&1; then
  echo "==> Installing blobfuse2 on host"
  sudo apt-get update
  sudo apt-get install -y lsb-release wget gnupg
  wget https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/packages-microsoft-prod.deb
  sudo dpkg -i packages-microsoft-prod.deb
  sudo apt-get update
  sudo apt-get install -y blobfuse2
fi
blobfuse2 --version

# Install python
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "==> Installing '$PYTHON_BIN' on host"
  sudo apt-get update
  sudo apt-get install -y "$PYTHON_BIN" python3-pip
fi
"$PYTHON_BIN" --version

# Mount Azure Blob Storage using blobfuse2
echo "==> Mounting Azure Blob on host"
mkdir -p "$AZURE_MOUNT_PATH"
echo "==> Updating /etc/fuse.conf"
sudo sed -i 's/^#user_allow_other/user_allow_other/' /etc/fuse.conf
"$PYTHON_BIN" "$HOST_REPO/mount_data.py" --mount-path="$AZURE_MOUNT_PATH"

echo "==> Pulling Docker image $IMAGE"
docker pull "$IMAGE"

echo "==> Running fresh container"
docker run --rm -it\
  --cap-add=SYS_ADMIN \
  --device=/dev/fuse \
  --security-opt apparmor:unconfined \
  --mount "type=bind,src=$HOST_REPO,dst=$CONTAINER_REPO" \
  --mount "type=bind,src=$AZURE_MOUNT_PATH,dst=$CONTAINER_AZURE_MOUNT_PATH" \
  -w "$CONTAINER_REPO" \
  "$IMAGE"

echo "==> Done"