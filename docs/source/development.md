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

Build CPU image:
```bash
docker build --target cpu -t skinet:cpu .
```

Build GPU image:
```bash
docker build --target gpu -t skinet:gpu .
```

Run container (example, bind-mount repo and host Azure mount):
```bash
docker run --rm \
  --cap-add=SYS_ADMIN \
  --device=/dev/fuse \
  --security-opt apparmor:unconfined \
  -v /path/on/host/repos/SkiNet:/workplace/SkiNet \
  -v /path/on/host/azure_blob_data:/mnt/data \
  -w /workplace/SkiNet \
  skinet:cpu
```

If you do not need FUSE inside the container (recommended): mount blobfuse on the VM host and only bind the mounted directory into the container; then you can omit the SYS_ADMIN/device flags.

## Lightning Studio

### Login to Codex

- Login wih ChatGPT credentials
- Authenticate wih ChatGPT as required, it will open a new window in your local browser. Note the port number
- Top right corner SSH, click on "Connect via SSH" and it will issue you with a connection string
- Modify it by adding relevant ports as follows for e.g. port 1455:
ssh -N -L 1455:localhost:1455 <user>@ssh.lightning.ai
