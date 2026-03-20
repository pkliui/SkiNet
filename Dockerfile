FROM ubuntu:22.04 AS base
# common setup for all stages except torch installs
ENV RUNNING_IN_DOCKER=true
ENV PIP_NO_CACHE_DIR=1

# Environment name and path to the project root directory
ARG ENV_NAME=skinet
ENV PROJECT_PATH=/workplace/SkiNet
# Append project path to the python path
ENV PYTHONPATH="${PYTHONPATH}:${PROJECT_PATH}"
# Specify work directory
WORKDIR ${PROJECT_PATH}
# Copy conda's environment file
COPY environment.yaml .
# specify mamba root -where environments will live i.e. /opt/micromamba/envs/skinet
ENV MAMBA_ROOT_PREFIX=/opt/micromamba

# install necessary packages
# DEBIAN_FRONTEND=noninteractive to prevent any additional prompts during the package installation
# rm -rf cleans up respective folder after installations
# build-essential is required to install conda
# lsb-release and blobfuse2 are required to mount Azure Blob Storage as a file system in Linux
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update && apt-get install -y --no-install-recommends \
    sudo wget curl unzip build-essential git lsb-release ca-certificates && \
    wget https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    apt-get update && apt-get install -y --no-install-recommends blobfuse2 && \
    rm -f packages-microsoft-prod.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install micromamba and create the environment
ARG MAMBA_URL=https://micro.mamba.pm/api/micromamba/linux-64/latest
RUN curl -Ls $MAMBA_URL | \
    tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba && \
    micromamba create -y -n skinet -f environment.yaml && \
    micromamba clean --all --yes && \
    rm -f environment.yaml && \
    rm -rf /root/.cache

# clean up installs and remove build-essential since it's not needed after conda installation
RUN apt-get purge -y wget curl unzip build-essential && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# the environment should be activated by default because micromamba was installed into /root/.local/bin
# add skinet's python executable to path
ENV PATH=/root/.local/bin:${MAMBA_ROOT_PREFIX}/envs/skinet/bin:${PATH}

# add a default user instead of using root in docker
#RUN useradd -ms /bin/bash skinet_runner
#USER skinet_runner

FROM base AS cpu
RUN micromamba run -n skinet python -m pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    micromamba run -n skinet python -m pip install --no-cache-dir torchdata && \
    rm -rf /root/.cache

FROM base AS gpu
RUN micromamba run -n skinet python -m pip install --no-cache-dir \
    torch torchvision torchaudio torchdata && \
    rm -rf /root/.cache