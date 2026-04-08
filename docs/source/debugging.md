# Instructions for debugging in Lightning Studio

- Setup: development is inside Docker container and the environment is set up and run via .lightning_studio/on_start.sh
- See Development section for more details
Variables:
```

# Image name on Docker Hub
IMAGE="pkliui/skinet:v4cpu"

# Determine a safe default for the home directory
DEFAULT_HOME="$HOME"

# Set repository variables
REPO_URL="${REPO_URL:-https://github.com/pkliui/SkiNet.git}"
HOST_REPO="${HOST_REPO:-$DEFAULT_HOME/repos/SkiNet}"
CONTAINER_REPO="${CONTAINER_REPO:-/workplace/SkiNet}"
BRANCH="${BRANCH:-train-proto}"

# Set Python binary
PYTHON_BIN="${PYTHON_BIN:-python3}"


# Data mount path on Lightning Storage
LIGHTNING_MOUNT_PATH="/teamspace/lightning_storage/"
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
```

**Example** for main_run.py file requiring --config coomand line argument (i.e. run as ```python main_run.py --config main_config.yaml```)

- Being in the Lightning studio's root (usually /teamspace/studios/this_studio), navigate to the repos root directory:  /teamspace/studios/this_studio/repos/SkiNet
- Go to the debugger by clicking on "Run and Debug"
- Select "Current python file with arguments" (being in the file you want to debug)
- Go to settings (the wheel icon), this will open a "launch.json" file. Specify "args": "--config ~/repos/SkiNet/debug_main_config.yaml"
- debug_main_config is the modified main_config.yaml *where the data path on Docker was replaced by the data path on Lightning Studio where we are debugging*: change   azure_blob_mount_point: "/mnt/data/" to azure_blob_mount_point: "/teamspace/lightning_storage/"
- Ensure you are on the repo root and click Run in Debugger
