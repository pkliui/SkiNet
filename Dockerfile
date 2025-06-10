FROM ubuntu:22.04
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

# install necessary packages
# DEBIAN_FRONTEND=noninteractive to prevent any additional prompts during the package installation
# rm -rf cleans up respective folder after installations
# build-essential is required to install conda
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update && apt-get install -y sudo wget curl unzip build-essential && \
    rm -rf /var/lib/apt/lists/*

# install AWS CLI
#RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
#    unzip awscliv2.zip && \
#    rm awscliv2.zip && \
#    ./aws/install

# install Azure CLI
# a separate ARG step to make this step cacheable
ARG AZURECLI_URL=https://aka.ms/InstallAzureCLIDeb
RUN curl -sL $AZURECLI_URL | sudo bash

# install conda and it to path
# a separate ARG step to make this step cacheable
# add conda_dir to path to make sure conda is available globally
ARG MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN wget $MINICONDA_URL -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ${CONDA_DIR}
ENV PATH=/root/.local/bin:${CONDA_DIR}/bin:${PATH}

# create a new environment and install packages
# conda init bash is required to make conda available in bash
# use mamba to speed up the process - conda often fails with connection failed error
RUN conda install -n base -c conda-forge mamba && \
    mamba env create -n skinet -f environment.yaml -y && rm environment.yaml && \
    conda init bash
# alternative way to create a conda environment, no mamba
#RUN conda init && conda env create -n skinet -f environment.yaml -y && rm environment.yaml

# activate the environment by default and add it to path
ENV CONDA_DEFAULT_ENV=${ENV_NAME}
RUN echo "conda activate ${ENV_NAME}" >> ~/.bashrc
ENV PATH=/opt/conda/envs/${ENV_NAME}/bin:${PATH}

# add a default user instead of using root in docker
#RUN useradd -ms /bin/bash skinet_runner
#USER skinet_runner

