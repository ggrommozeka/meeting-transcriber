#!/usr/bin/env bash
# Install host-side prerequisites for local (non-Docker) use.
#
# Tested on Ubuntu/Debian. The script never downloads model weights; run
# ./scripts/download_models.sh afterwards.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"
INSTALL_SYSTEM=1
INSTALL_TORCH=1

usage() {
  cat <<'EOF'
Usage: scripts/install_host_dependencies.sh [--venv PATH] [--no-system] [--no-torch]

  --venv PATH   virtualenv location (default: .venv)
  --no-system   do not install ffmpeg via apt
  --no-torch    assume torch is already installed
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) VENV="$2"; shift 2 ;;
    --no-system) INSTALL_SYSTEM=0; shift ;;
    --no-torch) INSTALL_TORCH=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required" >&2
  exit 1
fi

PY_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "Python: $PY_VERSION"
python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ is required")
PY

if [[ "$INSTALL_SYSTEM" == "1" ]]; then
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Installing FFmpeg..."
    if command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update
      sudo apt-get install -y ffmpeg
    else
      echo "Install ffmpeg manually for your distribution" >&2
      exit 1
    fi
  else
    echo "FFmpeg: $(ffmpeg -version | head -n1)"
  fi

  if ! command -v tesseract >/dev/null 2>&1; then
    echo "Installing Tesseract OCR (rus+eng)..."
    if command -v apt-get >/dev/null 2>&1; then
      sudo apt-get install -y tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng
    else
      echo "Install tesseract-ocr with rus/eng language data manually" >&2
    fi
  else
    echo "Tesseract: $(tesseract --version 2>/dev/null | head -n1)"
  fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA driver:"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
  echo "WARNING: nvidia-smi not found; GPU inference will not work. CPU is supported for testing only." >&2
fi

if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools

if [[ "$INSTALL_TORCH" == "1" ]]; then
  echo "Installing CUDA 12.4 PyTorch build..."
  pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
fi

pip install -r "$ROOT/requirements/asr.txt"
pip install -r "$ROOT/requirements/diarization.txt"
pip install -e "$ROOT"

echo
echo "Host environment ready. Activate it with: source $VENV/bin/activate"
echo "Next: ./scripts/download_models.sh && ./scripts/verify_models.sh"
