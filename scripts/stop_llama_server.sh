#!/usr/bin/env bash
# Stop the llama-server started by scripts/start_llama_server.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDFILE="$ROOT/logs/llama-server.pid"

if [[ ! -f "$PIDFILE" ]]; then
  echo "no pid file: $PIDFILE"
  exit 0
fi
PID="$(cat "$PIDFILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  for _ in $(seq 1 20); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 0.5
  done
  echo "stopped llama-server (pid $PID)"
else
  echo "process $PID is not running"
fi
rm -f "$PIDFILE"
