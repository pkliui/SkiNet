# Software development process


## Development environment

We use Ubuntu 22.04 inside a Docker container as our development environment. Please see more details on [developing inside a container using Visual Studio Code](https://code.visualstudio.com/docs/devcontainers/containers)

In short, one needs to make a Dockerfile, specifying the target OS and all necessary dependencies, make an image out of it and run it, specifying all necessary source and target volumes. For example,

```Docker
docker build -t skinet .
docker run --mount type=bind,src=/Users/Pavel/Documents/repos/SkiNet,dst=/workplace/SkiNet --mount type=bind,src=/Users/Pavel/.aws/,dst=/.aws/ -t skinet
```
Then, in Visual Studio Code, right click on the running container and select "Attach Visual Studio". This will open a new VSC window in the Docker container.