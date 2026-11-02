#!/usr/bin/env bash
# Build the Docker image. Models are not included in the image.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-meeting-transcriber:0.1.0}"

cd "$ROOT"
mkdir -p work data/output logs
docker build \
  --file docker/Dockerfile \
  --tag "$IMAGE" \
  "$@" \
  .
echo "Built $IMAGE"
