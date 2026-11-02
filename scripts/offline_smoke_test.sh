#!/usr/bin/env bash
# Offline smoke test: verifies models, then runs the pipeline with networking
# disabled where possible. Requires downloaded models and installed extras.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

AUDIO="${1:-$ROOT/data/input/smoke.wav}"

"$ROOT/scripts/verify_models.sh"

if [[ ! -f "$AUDIO" ]]; then
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg is required to generate the smoke fixture" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$AUDIO")"
  echo "Generating 5-second silence fixture: $AUDIO"
  ffmpeg -nostdin -hide_banner -loglevel error -y \
    -f lavfi -i anullsrc=r=16000:cl=mono -t 5 -c:a pcm_s16le "$AUDIO"
fi

RUNNER=(python3 -m meeting_transcriber.cli transcribe "$AUDIO"
  --config "$ROOT/config/default.yaml"
  --skip-diarization --skip-llm --force)

if command -v unshare >/dev/null 2>&1 && unshare -rn true 2>/dev/null; then
  echo "Running with a private network namespace (no network access)"
  unshare -rn "${RUNNER[@]}"
else
  echo "WARNING: network namespace isolation is unavailable; relying on offline env flags" >&2
  "${RUNNER[@]}"
fi

echo "Smoke test passed."
