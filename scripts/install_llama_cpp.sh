#!/usr/bin/env bash
# Install a pinned llama.cpp build with CUDA support into the user's home.
# No sudo, no system changes: binaries and CUDA runtime go under
# ~/.local/llama.cpp/<build>.
set -euo pipefail

BUILD="${LLAMA_CPP_BUILD:-b12032}"
CUDA_VARIANT="${LLAMA_CUDA_VARIANT:-13.4}"
DEST="${LLAMA_CPP_DIR:-$HOME/.local/llama.cpp/$BUILD}"
BASE_URL="https://github.com/ggml-org/llama.cpp/releases/download/${BUILD}"

if [[ -x "$DEST/llama-${BUILD}/llama-server" ]]; then
  echo "llama.cpp ${BUILD} already installed: $DEST/llama-${BUILD}/llama-server"
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$DEST"

echo "Downloading llama.cpp ${BUILD} (CUDA ${CUDA_VARIANT})..."
curl -L --fail --retry 2 \
  -o "$TMP/llama.tar.gz" \
  "${BASE_URL}/llama-${BUILD}-bin-ubuntu-cuda-${CUDA_VARIANT}-x64.tar.gz"
curl -L --fail --retry 2 \
  -o "$TMP/cudart.tar.gz" \
  "${BASE_URL}/cudart-llama-${BUILD}-bin-ubuntu-cuda-${CUDA_VARIANT}-x64.tar.gz"

tar -xzf "$TMP/llama.tar.gz" -C "$DEST"
tar -xzf "$TMP/cudart.tar.gz" -C "$DEST"

SERVER="$DEST/llama-${BUILD}/llama-server"
if [[ ! -x "$SERVER" ]]; then
  echo "installation failed: $SERVER not found" >&2
  exit 1
fi

echo "Installed: $SERVER"
"$SERVER" --version | head -n 2
echo
echo "Next: ./scripts/download_models.sh --only llm && ./scripts/start_llama_server.sh"
