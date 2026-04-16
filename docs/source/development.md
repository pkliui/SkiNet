# Software development process


## Development environment

We use Ubuntu 22.04 inside a Docker container as our development environment.
Micromamba  is used as a package management system and a conda environment is created using   ```environment.yaml``` inside Docker.

Please see more details on [developing inside a container using Visual Studio Code](https://code.visualstudio.com/docs/devcontainers/containers)

In short, one needs to make a Dockerfile, specifying the target OS and all necessary dependencies, make an image out of it and run it, specifying all necessary source and target volumes. Then, in Visual Studio Code, right click on the running container and select "Attach Visual Studio" or "Attach shell". This will open a new VSC window in the Docker container or attach Docker environment to your shell.


### Dockerfile overview

- base: Ubuntu 22.04 base image, installs system packages, blobfuse2 and micromamba, installs the conda environment from environment.yaml and exposes the project at /workplace/SkiNet.
- cpu: `FROM base AS cpu` — installs CPU PyTorch wheels into the micromamba environment.
- gpu: `FROM base AS gpu` — installs full (CUDA) PyTorch wheels.

Important notes
- micromamba is used to create the `skinet` conda environment. PATH is adjusted so the environment python is available.
- blobfuse2 is installed in the base image for convenience, but mounting Azure Blob Storage is usually done on the VM host and bind-mounted into containers.
- If you need to run inside the container with FUSE support, you must run the container with appropriate capabilities and devices (see examples below).

### Quick build & run commands

Build CPU image locally:
```bash
docker build --target cpu -t skinet:cpu .
```

Build GPU image locally:
```bash
docker build --target gpu -t skinet:gpu .
```

Run container (example, bind-mount repo and host Azure mount) locally:

```bash
docker run -it --mount type=bind,src=/Users/Pavel/Documents/repos/SkiNet,dst=/workplace/SkiNet skinet:cpu bash
```

to enable Azure Blob Fuse mounts, add
```
  --cap-add=SYS_ADMIN --device=/dev/fuse --security-opt apparmor:unconfined
```

If you do not need FUSE inside the container (recommended): mount blobfuse on the VM host and only bind the mounted directory into the container; then you can omit the SYS_ADMIN/device flags.

## Lightning Studio

### Build new Docker image

- Build CPU image locally, tag and push to Docker Hub to be able to launch from on_start.sh script later:

```bash
docker build --target cpu -t skinet:cpu .
docker tag skinet:cpu <dockerhub-username>/skinet:cpu
docker push <dockerhub-username>/skinet:cpu
```

OR equivalently tag immediately with the username as follows
```bash
docker build --target cpu -t <dockerhub-username>/skinet:cpu .
docker push <dockerhub-username>/skinet:cpu
```

### Set up the environment on Studio
- Being in the Lightning studio's root (usually /teamspace/studios/this_studio), run "on_start.sh" script that will
- 1. Clone the SkiNet repo (or update it if it already exists)
- 2. Pull the specified Docker image
- 3. Run a container from the image, mounting the repo and the Lightning Storage folder into the container

- **NOTE:** It is assumed that you uploaded your data into the Lightning Storage and that it is in folders
specified under PATH_ON_DATASTORE in repos/SkiNet/SkiNet/Azure/azure_settings.yaml

**Example on_start.sh**:
```bash
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

# This script is intended to be run on Lightning Studio. It will:
# 1. Clone the SkiNet repo (or update it if it already exists)
# 2. Pull the specified Docker image
# 3. Run a container from the image, mounting the repo and the Lightning Storage folder into the container

########################################################################################################

# NOTE: It is assumed that you uploaded your data into the Lightning Storage and that it is in folders
# specified under PATH_ON_DATASTORE in repos/SkiNet/SkiNet/Azure/azure_settings.yaml

########################################################################################################

set -Eeuo pipefail

# Set default values for environment variables if they are not already set

# Image name on Docker Hub
IMAGE="pkliui/skinet:v5cpu"

# Determine a safe default for the home directory
DEFAULT_HOME="$HOME"

# Set repository variables
REPO_URL="${REPO_URL:-https://github.com/pkliui/SkiNet.git}"
HOST_REPO="${HOST_REPO:-$DEFAULT_HOME/repos/SkiNet}"
CONTAINER_REPO="${CONTAINER_REPO:-/workplace/SkiNet}"
BRANCH="${BRANCH:-augm}"

# Set Python binary
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Data mount path on Lightning Storage
LIGHTNING_MOUNT_PATH="/teamspace/lightning_storage/ph2/"
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

echo "==> Running fresh container"
docker run --rm -it\
  --mount "type=bind,src=$HOST_REPO,dst=$CONTAINER_REPO" \
  --mount "type=bind,src=$LIGHTNING_MOUNT_PATH,dst=$CONTAINER_MOUNT_PATH" \
  -w "$CONTAINER_REPO" \
  "$IMAGE"

echo "==> Done. Docker is running. Attach Shell to the running container"
```

- In VSC "Containers" tab, attach the running container to the Bash shell.


### Debugging Cheatsheet

**See ports busy with non-docker processes**

- Install lsof
```bash
sudo apt-get update
sudo apt-get install lsof
```
- Example: port 6006
```bash
lsof -iTCP:6006 -sTCP:LISTEN -n -P
```

- Kill the process
```bash
kill <PID>
```



### Login to Codex

- Login wih ChatGPT credentials
- Authenticate wih ChatGPT as required, it will open a new window in your local browser. Note the port number
- Top right corner SSH, click on "Connect via SSH" and it will issue you with a connection string
- Modify it by adding relevant ports as follows for e.g. port 1455:
ssh -N -L 1455:localhost:1455 <user>@ssh.lightning.ai
