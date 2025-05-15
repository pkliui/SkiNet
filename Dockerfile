FROM ubuntu:22.04
ENV RUNNING_IN_DOCKER=true

ARG ENV_NAME=skinet
ENV PROJECT_PATH=/workplace/SkiNet
ENV CONDA_DIR=/opt/conda
COPY environment.yaml .
COPY pip_requirements.txt .

# install necessary packages
# DEBIAN_FRONTEND=noninteractive to prevent any additional prompts to further customize installation options
# rm -rf cleans up respective folder after installations
# build-essential is required to install conda
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update && apt-get install -y sudo wget curl unzip build-essential && \
    rm -rf /var/lib/apt/lists/*

# install conda and it to path
# a separate ARG step to make this step cacheable
ARG MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN wget $MINICONDA_URL -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ${CONDA_DIR}
ENV PATH=/root/.local/bin:${CONDA_DIR}/bin:${PATH}s

# create a new environment and install packages
# conda init bash is required to make conda available in bash
# use mamba to speed up the process - conda often fails with connection failed error
RUN conda install -n base -c conda-forge mamba && \
    mamba env create -n skinet -f environment.yaml -y && rm environment.yaml && \
    conda init bash
# alternative way to create a conda environment, no mamba
#RUN conda init && conda env create -n skinet -f environment.yaml -y && rm environment.yaml

# add pip dependencies
# the reason for having a separate requirements file for pip is that conda fails with "conflicting versions" error if pip packages are installed from yaml
# don't install for the moment due to version conflics in Azure
#RUN pip install -r pip_requirements.txt && rm pip_requirements.txt

# activate the environment by default
ENV CONDA_DEFAULT_ENV=${ENV_NAME}
RUN echo "conda activate ${ENV_NAME}" >> ~/.bashrc
# LD_LIBRARY_PATH fixing "import torch" error https://github.com/pytorch/pytorch/issues/111469, 
# alongside with mamba install python pytorch torchvision torchdata (specified in environment.yaml without any fixed versions)
ENV LD_LIBRARY_PATH=/opt/conda/envs/skinet/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
# add conda environment to path
ENV PATH=/opt/conda/envs/${ENV_NAME}/bin:${PATH}

# add project's directory to python's path
ENV PYTHONPATH="${PYTHONPATH}:${PROJECT_PATH}"
WORKDIR /workplace/SkiNet

# install AWS CLI
#RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
#    unzip awscliv2.zip && \
#    rm awscliv2.zip && \
#    ./aws/install

# install Azure CLI
# required for authentication ?
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash 

# add a default user instead of using root in docker
#RUN useradd -ms /bin/bash skinet_runner
#USER skinet_runner

