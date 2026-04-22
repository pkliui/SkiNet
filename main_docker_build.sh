# This script builds a Docker image in the current directory
# Please modify --target and IMAGE_TAG accordingly!

ENV_HASH=$(sha256sum environment.yaml | cut -c1-64)
IMAGE_TAG="pkliui/skinet:v9gpu"
docker build \
  --build-arg ENV_HASH=$ENV_HASH \
  --target gpu \
  -t $IMAGE_TAG .
docker push $IMAGE_TAG
