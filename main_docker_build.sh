# This script builds a Docker image in the current directory

ENV_HASH=$(sha256sum environment.yaml | cut -c1-64)
docker build \
  --build-arg ENV_HASH=$ENV_HASH \
  --target gpu \
  -t pkliui/skinet:v9gpu .
