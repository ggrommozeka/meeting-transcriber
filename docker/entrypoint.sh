#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  set -- --help
fi

case "$1" in
  meeting-transcriber)
    shift
    exec meeting-transcriber "$@"
    ;;
  transcribe|verify-models|list-models|download-models|validate-config|build-glossary|extract-audio|speakers|label-speakers|report|--help|-h|--version)
    exec meeting-transcriber "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
