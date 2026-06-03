# Software development process

## Development environment

We use Ubuntu 22.04 inside a Docker container as our development environment. Micromamba is used as a package management system and an environment is created using   ```environment.yaml``` inside Docker.

Please see more details on [developing inside a container using Visual Studio Code](https://code.visualstudio.com/docs/devcontainers/containers)

## Docker

### Dockerfile overview

- base: Ubuntu 22.04 base image, installs system packages, blobfuse2 and micromamba, installs the conda environment from environment.yaml and exposes the project at /workplace/SkiNet.
- cpu: `FROM base AS cpu` — installs CPU PyTorch wheels into the micromamba environment.
- gpu: `FROM base AS gpu` — installs full (CUDA) PyTorch wheels.

Important notes
- micromamba is used to create the `skinet` conda environment. PATH is adjusted so the environment python is available.
- blobfuse2 is installed in the base image for convenience, but mounting Azure Blob Storage is usually done on the VM host and bind-mounted into containers.
- If you need to run inside the container with FUSE support, you must run the container with appropriate capabilities and devices (see examples below)
- images are labelled with the environment's hash

### Ways to build and run the containers

*** Build and push CPU and GPU images to the Hub ***

```bash main_docker_build.sh```

*** Manually build and push images to the Hub ***

- To build and push an image to the Hub, run the following:

```bash
ENV_HASH=$(sha256sum environment.yaml | cut -c1-64)
IMAGE_TAG="pkliui/skinet:v9gpu"   # adjust tag to match the version in on_start_gpu.sh / on_start_cpu.sh
docker build \
  --build-arg ENV_HASH=$ENV_HASH \
  --target gpu \
  -t $IMAGE_TAG .
docker push $IMAGE_TAG
```

*** Quick build & run commands ***
- For quick experimenting and debugging purposes, use the following commands:

Build CPU image locally:
```bash
docker build --target cpu -t skinet:cpu .
```

Build GPU image locally:
```bash
docker build --target gpu -t skinet:gpu .
```

Run container bind-mount repo locally:
```bash
docker run -it --mount type=bind,src=/Users/Pavel/Documents/repos/SkiNet,dst=/workplace/SkiNet skinet:cpu bash
```

### Additional options

- To enable Azure Blob Fuse mounts, add
```
  --cap-add=SYS_ADMIN --device=/dev/fuse --security-opt apparmor:unconfined
```

If you do not need FUSE inside the container (recommended): mount blobfuse on the VM host and only bind the mounted directory into the container; then you can omit the SYS_ADMIN/device flags.


## Lightning Studio

### Set up the environment on Studio

The following startup scripts are provided (each clones/updates the repo, pulls the Docker image, and launches a container):

| Script | Image | Use case |
|---|---|---|
| `on_start_gpu.sh` | `pkliui/skinet:v9gpu` | GPU training on a Lightning Studio GPU instance |
| `on_start_cpu.sh` | `pkliui/skinet:v9cpu` | CPU development / debugging |


**NOTE:** When running code on Lightning Studio, data must be placed in Lightning Storage. The startup scripts bind-mount it into the container at `/mnt/data/`. Set `azure_data: False` and `local_data_root: "/mnt/data/"` in `main_config.yaml`.

#### CPU modes (`on_start_cpu.sh`)

| MODE | What runs |
|---|---|
| `interactive` | Drops into a `bash` shell inside the container (default) |
| `test` | `python3 main_run.py --config main_config.yaml --test-only --checkpoint <CHECKPOINT>` |

The scripts bootstrap Docker and dispatch to the Python entry points. For what those entry points do — config fields, sweep mechanics, callbacks, reproducibility — see [training.md](training.md).

#### GPU modes (`on_start_gpu.sh`)

> **`RUN_TRAINING` defaults to `false`** (dry-run guard). Set `RUN_TRAINING=true` to actually pull the image and launch the container. Omitting it will print the resolved command and exit without running anything.

| MODE | What runs |
|---|---|
| `train` | `python main_run.py --config main_config.yaml` |
| `seeds` | `python run_seeds.py --config main_config.yaml --seeds <SEEDS> [--encoder-modes ...] [--merge-modes ...]` |
| `sweep` | `python optuna_sweep.py --config main_config.yaml` (monitor/direction from `SWEEP_CONFIG` in YAML) |
| `test` | `python main_run.py --config main_config.yaml --test-only --checkpoint <CHECKPOINT>` |
| `calibrate` | `python calibrate_threshold.py --config main_config.yaml` |

#### Run CPU on Studio

```bash
# Interactive shell — ph2 dataset
DATASET=ph2 MODE=interactive bash on_start_cpu.sh

# Interactive shell — isic2017 dataset (default)
MODE=interactive bash on_start_cpu.sh

# Test a checkpoint — isic2017
MODE=test CHECKPOINT=runs/exp1/best.ckpt bash on_start_cpu.sh

# Test a checkpoint — ph2
DATASET=ph2 MODE=test CHECKPOINT=runs/exp1/best.ckpt bash on_start_cpu.sh
```

#### Run GPU on Studio

```bash
# Single training run — isic2017 dataset (default)
RUN_TRAINING=true MODE=train bash on_start_gpu.sh

# Single training run — ph2 dataset
RUN_TRAINING=true DATASET=ph2 MODE=train bash on_start_gpu.sh

# Multi-seed run using seeds/encoder/merge modes from YAML config
RUN_TRAINING=true MODE=seeds SEEDS="1 2 3 4 5" bash on_start_gpu.sh

# Multi-seed run with explicit encoder and merge modes (overrides YAML)
RUN_TRAINING=true DATASET=isic2017 ENCODER_MODES="classical" MERGE_MODES="classical local_refinement he2 attention_gate" MODE=seeds SEEDS="100 101 102 103 104" bash on_start_gpu.sh

# Optuna sweep — monitor and direction read from SWEEP_CONFIG in the YAML (recommended)
RUN_TRAINING=true MODE=sweep bash on_start_gpu.sh

# Optuna sweep — one-off override without editing the YAML
RUN_TRAINING=true MODE=sweep SWEEP_MONITOR=perf/samples_per_sec SWEEP_DIRECTION=maximize bash on_start_gpu.sh

# Threshold calibration
RUN_TRAINING=true MODE=calibrate bash on_start_gpu.sh

# Dry run — build command but skip container launch
RUN_TRAINING=false MODE=train bash on_start_gpu.sh

# Release GPU after training (default: NOT released)
RELEASE_GPU=true RUN_TRAINING=true MODE=train bash on_start_gpu.sh
```

After the container starts, attach to it in VSCode via the "Containers" tab → "Attach Shell".


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

### Force save file from command line
sudo tee file_name.py > /dev/null << 'EOF'
FILE-CONTENT
EOF

### Login to Codex

- Login wih ChatGPT credentials
- Authenticate wih ChatGPT as required, it will open a new window in your local browser. Note the port number
- Top right corner SSH, click on "Connect via SSH" and it will issue you with a connection string
- Modify it by adding relevant ports as follows for e.g. port 1455:
ssh -N -L 1455:localhost:1455 <user>@ssh.lightning.ai

## Kaggle notebooks

| `on_start_kaggle.sh` | — | Kaggle notebook environment, for e.g. GPU training |
