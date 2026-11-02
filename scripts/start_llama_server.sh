#!/usr/bin/env bash
# Start llama-server for the local meeting-report model.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BUILD="${LLAMA_CPP_BUILD:-b12032}"
BIN="${LLAMA_SERVER:-$HOME/.local/llama.cpp/$BUILD/llama-$BUILD/llama-server}"
MODEL="${LLM_MODEL_PATH:-$ROOT/models/qwen3.5-9b-gguf/Qwen3.5-9B-UD-Q4_K_XL.gguf}"
HOST="${LLAMA_HOST:-127.0.0.1}"
PORT="${LLAMA_PORT:-8080}"
CTX="${LLAMA_CTX:-8192}"
PIDFILE="$ROOT/logs/llama-server.pid"
LOG="$ROOT/logs/llama-server.log"

mkdir -p "$ROOT/logs"

if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
  echo "llama-server already running on http://${HOST}:${PORT}"
  exit 0
fi

if [[ ! -x "$BIN" ]]; then
  echo "llama-server not found: $BIN" >&2
  echo "Run ./scripts/install_llama_cpp.sh" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "model not found: $MODEL" >&2
  echo "Run ./scripts/download_models.sh --only llm" >&2
  exit 1
fi

echo "Starting llama-server: $MODEL"
nohup "$BIN" \
  -m "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  -ngl 99 \
  -c "$CTX" \
  -b 2048 \
  -ub 512 \
  --parallel 1 \
  --reasoning off \
  --jinja \
  > "$LOG" 2>&1 &
echo $! > "$PIDFILE"

for _ in $(seq 1 120); do
  if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    echo "llama-server ready: http://${HOST}:${PORT}/v1 (pid $(cat "$PIDFILE"))"
    echo "log: $LOG"
    exit 0
  fi
  sleep 1
done

echo "llama-server did not become healthy; see $LOG" >&2
tail -n 20 "$LOG" >&2
exit 1
