# CLI reference

```text
meeting-transcriber transcribe <audio-or-video> [options]
meeting-transcriber transcribe -i <audio-or-video> [options]
meeting-transcriber transcribe-video -i <video> [options]
meeting-transcriber extract-audio <media> [options]
meeting-transcriber speakers <transcript.json> [options]
meeting-transcriber label-speakers <transcript.json> --map <speakers.yaml>
meeting-transcriber report <transcript.json> [options]
meeting-transcriber verify-models [--only all|asr|diarization|llm]
meeting-transcriber list-models [--only ...]
meeting-transcriber download-models [--only ...] [--force]
meeting-transcriber validate-config [--config PATH]
meeting-transcriber build-glossary <project-path> [options]
```

`transcribe` and `transcribe-video` accept `-i/--input` (a positional path still
works). If `--output-dir` is omitted, the transcript is written next to the
input file. `transcribe-video` enables `visual` and skips the LLM report.

Python API:

```python
from meeting_transcriber import transcribe_audio, transcribe_video

transcribe_audio("meeting.mp4")            # no video analysis
transcribe_video("meeting.mp4")            # OCR of slides/code
transcribe_video("meeting.mp4", output_dir="out")
```

Global options accepted by subcommands that use configuration:

| Option | Meaning |
|---|---|
| `--config PATH` | YAML configuration (default: `config/default.yaml`) |
| `--models-dir DIR` | override `paths.models_dir` |

## transcribe

```bash
meeting-transcriber transcribe data/input/meeting.webm \
  --config config/default.yaml \
  --output-dir data/output
```

| Option | Meaning |
|---|---|
| `-i, --input FILE` | input audio/video (positional path also accepted) |
| `--language ru` | explicit ASR language |
| `--output-dir DIR` | override `paths.output_dir` |
| `--profile NAME` | use one terminology profile |
| `--num-speakers N` | known participant count (oracle mode) |
| `--max-speakers N` | upper bound for automatic counting (default 20) |
| `--skip-diarization` | ASR only |
| `--skip-llm` | do not generate the report (default for the simple commands) |
| `--llm` | generate the LLM report (requires the local llama-server) |
| `--speaker-map FILE` | apply display names before report/output |
| `--report-style STYLE` | `asd-ste100` (short) or `plain` (full report) |
| `--device cuda\|cpu\|auto` | override device |
| `--audio-stream N` | 0-based audio track index (FFmpeg `0:a:N`) |
| `--visual/--no-visual` | enable optional OCR of slides/code (off by default) |
| `--keep-frames/--no-keep-frames` | Keep frames of code cards next to the report |
| `--ocr-backend tesseract` | Local OCR backend for visual analysis |
| `--force` | ignore cached stages and overwrite outputs |

## extract-audio

```bash
meeting-transcriber extract-audio data/input/meeting.mp4 \
  --output data/output/meeting.mka --audio-stream 1 --force
```

Supported output containers: `.mka` (recommended, stream copy), `.flac`, `.wav`.

## speakers

```bash
meeting-transcriber speakers data/output/meeting.json
```

Prints talk time, segment count and sample quotes, and writes
`data/output/meeting.speakers.yaml` with empty `name` fields.

| Option | Meaning |
|---|---|
| `--template PATH` | write the template elsewhere |
| `--force` | overwrite an existing template |

## label-speakers

```bash
meeting-transcriber label-speakers data/output/meeting.json \
  --map data/output/meeting.speakers.yaml
```

Updates `display_name` in JSON and regenerates Markdown/SRT. Does not re-run ASR
or diarization.

## report

```bash
./scripts/start_llama_server.sh
meeting-transcriber report data/output/meeting.json \
  --map data/output/meeting.speakers.yaml
./scripts/stop_llama_server.sh
```

| Option | Meaning |
|---|---|
| `--map FILE` | apply speaker names before generation |
| `--report-style STYLE` | `asd-ste100` (short) or `plain` (full report) |
| `--visual/--no-visual` | use cached OCR screen cards from `work/<meeting>/visual.json` |

Writes `<name>.report.json`, `<name>.report.md`, and appends the report to
`<name>.markdown`.

## Model commands

```bash
meeting-transcriber download-models --only llm
meeting-transcriber verify-models
meeting-transcriber list-models
```

`--only` accepts `all`, `asr`, `diarization`, `llm`. `download-models` is the
only command that accesses the network for weights. Without `--only` on a
terminal it shows an interactive table with model sizes and presence, then asks
which groups to download; use `--only all` for non-interactive full download.
Downloads show per-file progress bars and verify SHA-256 hashes.

## validate-config

```bash
meeting-transcriber validate-config --config config/default.yaml
```

Prints the effective configuration and exits non-zero on invalid keys or values.

## build-glossary

```bash
meeting-transcriber build-glossary /path/to/cpp-project \
  --output config/terminology/generated.yaml \
  --profile generated
```

Scans an explicitly provided C++/CMake tree (classes, structs, namespaces,
types, functions, CMake targets). It never scans a path you did not pass.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | pipeline/runtime error |
| 2 | configuration error |
| 3 | model missing or checksum mismatch |
| 4 | GPU unavailable/incompatible |

## Output behaviour

- Existing outputs are not overwritten without `--force`; a timestamp suffix is
  added instead.
- `work/<meeting>/` stores stage artifacts and resume state.
- Re-running without `--force` reuses completed stages.
- Configuration changes invalidate only the affected stages.
