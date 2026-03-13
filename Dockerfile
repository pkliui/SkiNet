FROM ubuntu:22.04 AS base
# common setup for all stages except torch installs
ENV RUNNING_IN_DOCKER=true

# Specify environment variables for the environment name, paths to the project root directory and conda
# Append project path to the python path
# Specify work directory
# Copy conda's environment file
ARG ENV_NAME=skinet
ENV PROJECT_PATH=/workplace/SkiNet
ENV CONDA_DIR=/opt/conda
ENV PYTHONPATH="${PYTHONPATH}:${PROJECT_PATH}"
WORKDIR ${PROJECT_PATH}
COPY environment.yaml .

# no pip cache
ENV PIP_NO_CACHE_DIR=1

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

# install conda and it to path
# a separate ARG step to make this step cacheable
# add conda_dir to path to make sure conda is available globally
ARG MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN wget $MINICONDA_URL -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ${CONDA_DIR} && \
    rm -f ~/miniconda.sh
ENV PATH=/root/.local/bin:${CONDA_DIR}/bin:${PATH}


RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# create a new environment and install packages
# conda init bash is required to make conda available in bash
# use mamba to speed up the process - conda often fails with connection failed error
RUN conda install -n base -c conda-forge mamba && \
    mamba env create -n skinet -f environment.yaml -y && \
    conda clean -a -f -y && \
    rm environment.yaml && \
    conda init bash
# alternative way to create a conda environment, no mamba
#RUN conda init && conda env create -n skinet -f environment.yaml -y && rm environment.yaml

# clean up installs and remove build-essential since it's not needed after conda installation
RUN apt-get purge -y sudo wget curl unzip build-essential && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# activate the environment by default and add it to path
ENV CONDA_DEFAULT_ENV=${ENV_NAME}
RUN echo "conda activate ${ENV_NAME}" >> ~/.bashrc
ENV PATH=/opt/conda/envs/${ENV_NAME}/bin:${PATH}

# add a default user instead of using root in docker
#RUN useradd -ms /bin/bash skinet_runner
#USER skinet_runner

FROM base AS cpu
RUN conda run -n skinet python -m pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    conda run -n skinet python -m pip install --no-cache-dir torchdata && \
    rm -rf /root/.cache

FROM base AS gpu
RUN conda run -n skinet python -m pip install --no-cache-dir \
    torch torchvision torchaudio torchdata && \
    rm -rf /root/.cache