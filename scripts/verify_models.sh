#!/usr/bin/env bash
# Verify that local model files exist, have the expected size and SHA-256.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

exec python3 -m meeting_transcriber.cli verify-models --config "$ROOT/config/default.yaml" "$@"
