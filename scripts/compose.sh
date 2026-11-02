#!/usr/bin/env bash
# docker compose wrapper: exports host UID/GID and creates writable mounts.
#
# Usage:
#   ./scripts/compose.sh run --rm transcriber transcribe /data/input/meeting.webm \
#     --config /app/config/docker.yaml --skip-llm
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"
mkdir -p "$ROOT/work" "$ROOT/data/output" "$ROOT/logs"

exec docker compose -f "$ROOT/compose.yaml" "$@"
