#!/usr/bin/env bash
# Download model weights to the local models/ directory.
# This is the only command that touches the network for model weights.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec python3 -m meeting_transcriber.cli download-models --config "$ROOT/config/default.yaml" "$@"
