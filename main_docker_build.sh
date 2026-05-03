# This script builds two Docker images for GPU and CPU
# Dockerfile is expected to be in the current directory,
# along with environment.yaml file
# Please modify the version in both IMAGE_TAGs accordingly!

ENV_HASH=$(sha256sum environment.yaml | cut -c1-64)
IMAGE_TAG_GPU="pkliui/skinet:v9gpu"
IMAGE_TAG_CPU="pkliui/skinet:v9cpu"

docker build \
  --build-arg ENV_HASH=$ENV_HASH \
  --target gpu \
  -t $IMAGE_TAG_GPU .
docker push $IMAGE_TAG_GPU

docker build \
  --build-arg ENV_HASH=$ENV_HASH \
  --target cpu \
  -t $IMAGE_TAG_CPU .
docker push $IMAGE_TAG_CPU
