# Software development process


## Development environment

We use Ubuntu 22.04 inside a Docker container as our development environment. Please see more details on [developing inside a container using Visual Studio Code](https://code.visualstudio.com/docs/devcontainers/containers)

In short, one needs to make a Dockerfile, specifying the target OS and all necessary dependencies, make an image out of it and run it, specifying all necessary source and target volumes. For example,

```Docker
docker build -t skinet .
docker run --mount type=bind,src=/Users/Pavel/Documents/repos/SkiNet,dst=/workplace/SkiNet --mount type=bind,src=/Users/Pavel/.aws/,dst=/.aws/ -t skinet
```
Then, in Visual Studio Code, right click on the running container and select "Attach Visual Studio". This will open a new VSC window in the Docker container.

## Conda Environment and Dependencies

- SkiNet uses Miniconda as a package management system 
- A conda environment is created using   ```environment.yaml``` 

- Pytorch packages are being installed through pip as Pytorch's official Anaconda channel was [deprecated](https://github.com/pytorch/pytorch/issues/138506) (with 2.5 being the last release on anaconda)
 
- [Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-access-data-interactive?view=azureml-api-2&tabs=adls) requires the latest azure-fsspec, mltable, and azure-ai-ml python libraries  that are also installed through pip

##  Dockerfile

- We use Ubuntu 22.04 image and are running inside the container. Dockerfile has the following content:

```
FROM ubuntu:22.04
ENV RUNNING_IN_DOCKER=true
```

- Specify environment variables for the environment name, paths to the project root directory and conda
- Append project path to the python path
- Specify work directory
- Copy conda's environment file 
  
```
ARG ENV_NAME=skinet
ENV PROJECT_PATH=/workplace/SkiNet
ENV CONDA_DIR=/opt/conda
ENV PYTHONPATH="${PYTHONPATH}:${PROJECT_PATH}"
WORKDIR ${PROJECT_PATH}
COPY environment.yaml .
```


- Install necessary utilities to be able to run commands with elevated privileges, download files, unzip and build tools for conda
```
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update && apt-get install -y sudo wget curl unzip build-essential && \
    rm -rf /var/lib/apt/lists/*
```


- Install Azure CLI
```
ARG AZURECLI_URL=https://aka.ms/InstallAzureCLIDeb
RUN curl -sL $AZURECLI_URL | sudo bash
```


- Install Miniconda and add its binaries to the system PATH so that conda is globally accessible, for e.g. the following step below or calling "conda list" from docker:
```
ARG MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN wget $MINICONDA_URL -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ${CONDA_DIR}
ENV PATH=/root/.local/bin:${CONDA_DIR}/bin:${PATH}
```



- Create a new conda environment from file ```environment.yaml``` using Mamba
```
RUN conda install -n base -c conda-forge mamba && \
    mamba env create -n skinet -f environment.yaml -y && rm environment.yaml && \
    conda init bash
```


- Activate the conda environment
- Add "bin" directory of the conda environment to the PATH to be able to call binaries installed in the environment directly 
```
ENV CONDA_DEFAULT_ENV=${ENV_NAME}
RUN echo "conda activate ${ENV_NAME}" >> ~/.bashrc
ENV PATH=/opt/conda/envs/${ENV_NAME}/bin:${PATH}
```






