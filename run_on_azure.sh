#!/usr/bin/env bash
set -Eeuo pipefail

IMAGE="${1:?Usage: $0 <docker-image> [script args...]}"
shift || true

DEFAULT_HOME="${HOME:-/home/${USER:-$(whoami)}}"

REPO_URL="${REPO_URL:-https://github.com/pkliui/SkiNet.git}"
HOST_REPO="${HOST_REPO:-$DEFAULT_HOME/repos/SkiNet}"
CONTAINER_REPO="${CONTAINER_REPO:-/workplace/SkiNet}"
BRANCH="${BRANCH:-data_on_azure}"

PYTHON_BIN="${PYTHON_BIN:-python}"

USE_MANAGED_IDENTITY="${USE_MANAGED_IDENTITY:-true}"
AZURE_MANAGED_IDENTITY_CLIENT_ID="${AZURE_MANAGED_IDENTITY_CLIENT_ID:-}"

AZURE_MOUNT_PATH="${AZURE_MOUNT_PATH:-/mnt/azure_blob_data}"
CONTAINER_AZURE_MOUNT_PATH="${CONTAINER_AZURE_MOUNT_PATH:-/mnt/data}"

echo "Running with the following configuration:"
echo "DEFAULT_HOME=$DEFAULT_HOME"
echo "REPO_URL=$REPO_URL"
echo "HOST_REPO=$HOST_REPO"
echo "CONTAINER_REPO=$CONTAINER_REPO"
echo "BRANCH=$BRANCH"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "USE_MANAGED_IDENTITY=$USE_MANAGED_IDENTITY"
echo "AZURE_MANAGED_IDENTITY_CLIENT_ID=$AZURE_MANAGED_IDENTITY_CLIENT_ID"
echo "AZURE_MOUNT_PATH=$AZURE_MOUNT_PATH"
echo "CONTAINER_AZURE_MOUNT_PATH=$CONTAINER_AZURE_MOUNT_PATH"


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
python "$HOST_REPO/scripts/mount_blob_on_host.py"


echo "==> Mounting Azure Blob on host"
python "$HOST_REPO/mount_data.py"


if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not installed"
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is not installed"
  exit 1
fi


HOST_REPO_PARENT="$(dirname "$HOST_REPO")"
mkdir -p "$HOST_REPO_PARENT"

if [[ -d "$HOST_REPO/.git" ]]; then
  echo "==> Repo already exists, updating it"

  if [[ -n "${HOME:-}" ]]; then
    git config --global --add safe.directory "$HOST_REPO" || true
  else
    echo "WARNING: HOME is not set; skipping git config --global safe.directory"
  fi

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

echo "==> Running fresh container"


docker run --rm \
  --cap-add=SYS_ADMIN \
  --device=/dev/fuse \
  --security-opt apparmor:unconfined \
  -e "USE_MANAGED_IDENTITY=$USE_MANAGED_IDENTITY" \
  --mount "type=bind,src=$HOST_REPO,dst=$CONTAINER_REPO" \
  --mount "type=bind,src=$AZURE_MOUNT_PATH,dst=$CONTAINER_AZURE_MOUNT_PATH" \
  -w "$CONTAINER_REPO" \
  "$IMAGE"

echo "==> Done"