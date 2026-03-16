#!/usr/bin/env bash
set -Eeuo pipefail

IMAGE="${1:?Usage: $0 <docker-image> [script args...]}"
shift || true

REPO_URL="${REPO_URL:-https://github.com/pkliui/SkiNet.git}"
#HOST_REPO="${HOST_REPO:-/mnt/batch/tasks/shared/LS_root/mounts/clusters/skinet-compute/code/Users/pavel.kliuiev/repos/SkiNet}"
CONTAINER_REPO="${CONTAINER_REPO:-/workplace/SkiNet}"
BRANCH="${BRANCH:-data_on_azure}"
PYTHON_BIN="${PYTHON_BIN:-python}"
USE_MANAGED_IDENTITY="${USE_MANAGED_IDENTITY:-true}"
AZURE_MANAGED_IDENTITY_CLIENT_ID="${AZURE_MANAGED_IDENTITY_CLIENT_ID:-}"

DEFAULT_HOME="${HOME:-/home/${USER:-$(whoami)}}"
HOST_REPO="${HOST_REPO:-$DEFAULT_HOME/repos/SkiNet}"

echo "IMAGE=$IMAGE"
echo "REPO_URL=$REPO_URL"
echo "HOST_REPO=$HOST_REPO"
echo "CONTAINER_REPO=$CONTAINER_REPO"
echo "BRANCH=$BRANCH"
echo "USE_MANAGED_IDENTITY=$USE_MANAGED_IDENTITY"


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
  -w "$CONTAINER_REPO" \
  "$IMAGE"

echo "==> Done"