# meeting-transcriber

Fully local transcription of meetings with Russian speech and mixed
Russian/English technical terminology, speaker diarization for roughly 8–15
participants, terminology correction, and an optional local-LLM meeting report.
Video input can additionally provide OCR context from slides and code shown on
screen.

No cloud services, no Hugging Face token at runtime, and no model downloads
during inference.

## Key properties

- **Two simple entry points** — audio-only transcription and video-aware
  transcription with screen OCR.
- **`-i/--input`** — the input file is passed explicitly; the transcript is
  written next to it by default.
- **Optional video analysis** — off by default; enable it per run or in the
  config. Stable screen states are OCR'd, code is classified and linked from
  the meeting report.
- **Report with code references** — items link to `C<n>` cards with timestamps,
  OCR text and frame images.
- **Local ASR** — faster-whisper / Whisper large-v3-turbo, CTranslate2.
- **Diarization without a 4-speaker cap** — Silero VAD → TitaNet-Large → NeMo
  spectral clustering, up to 20 speakers.
- **Auditable corrections** — raw ASR text is preserved; every terminology
  replacement is recorded.
- **Fully offline runtime** — weights are local files; network is used only by
  explicit model-download commands.
- **8 GB VRAM target** — ASR and diarization run sequentially on one GPU.
- **Rich CLI** — live per-stage progress, a result panel, and interactive model
  selection.

## Quick start

Requirements: Linux, Python 3.10+, FFmpeg, an NVIDIA GPU with 8 GB+ VRAM
(CPU works for testing only). Video analysis also needs Tesseract with Russian
and English language data.

```bash
./scripts/install_host_dependencies.sh
source .venv/bin/activate
./scripts/download_models.sh        # interactive on a terminal
./scripts/verify_models.sh
```

Get a transcript next to the input file:

```bash
# Audio only: no video analysis, no LLM report
meeting-transcriber transcribe -i /path/to/meeting.mp4

# Audio + OCR of slides and code shown in the video
meeting-transcriber transcribe-video -i /path/to/meeting.mp4
```

Both commands accept a positional path too, but `-i` is the recommended form:

```bash
meeting-transcriber transcribe /path/to/meeting.mp4   # still supported
```

The same two operations are available from Python:

```python
from meeting_transcriber import transcribe_audio, transcribe_video

result = transcribe_audio("/path/to/meeting.mp4")
print(result.transcript.meeting_id)

result = transcribe_video(
    "/path/to/meeting.mp4",
    output_dir="/path/to/transcripts",   # optional
)
```

`transcribe_audio()` and `transcribe_video()` skip the LLM report.

## Control and modes

### 1. Audio-only transcription

This is the default for `transcribe`:

```bash
meeting-transcriber transcribe -i meeting.mp4
```

- video frames are ignored;
- the LLM report is not generated;
- outputs go next to `meeting.mp4` unless `--output-dir DIR` is given.

Useful options:

```bash
--output-dir DIR        # write outputs elsewhere
--language ru           # ASR language
--skip-diarization      # ASR only
--num-speakers N        # exact speaker count
--max-speakers N        # upper bound for automatic counting
--audio-stream N        # select a 0-based audio track
--keep-frames           # relevant only with video analysis
```

### 2. Video-aware transcription

`transcribe-video` enables the visual stage and keeps the LLM report disabled:

```bash
meeting-transcriber transcribe-video -i meeting.mp4
```

The same can be done with `transcribe --visual`:

```bash
meeting-transcriber transcribe -i meeting.mp4 --visual
```

Video analysis is **off by default**. Explicit controls:

| Command / config | Effect |
|---|---|
| `transcribe-video -i FILE` | video OCR on |
| `transcribe -i FILE --visual` | video OCR on |
| `transcribe -i FILE --no-visual` | video OCR off, even if enabled in config |
| `visual.enabled: true` | enable video OCR by default in the config |
| `--keep-frames / --no-keep-frames` | save or skip code-card frames |

Video analysis requires the `tesseract` executable. The host setup script
installs `tesseract-ocr`, `tesseract-ocr-rus` and `tesseract-ocr-eng`. For a
fully offline setup, point `visual.tessdata_dir` at a directory with
`rus.traineddata` and `eng.traineddata`.

### 3. LLM report

The report is separate from transcription and requires the local llama.cpp
server:

```bash
./scripts/install_llama_cpp.sh          # once
./scripts/start_llama_server.sh

meeting-transcriber report /path/to/meeting.json

./scripts/stop_llama_server.sh
```

If the transcript was produced with `--visual`, the report uses the cached
screen cards from `work/<meeting_id>/visual.json`. Disable that with
`--no-visual`.

There is also a convenience flag, `--llm`, that generates the report in the same
run:

```bash
meeting-transcriber transcribe -i meeting.mp4 --visual --llm
```

On an 8 GB GPU, prefer the two-step workflow: stop `llama-server`, transcribe,
then start `llama-server` and generate the report.

## Setup and models

```bash
./scripts/install_host_dependencies.sh   # FFmpeg, Tesseract, Python, CUDA, NeMo
source .venv/bin/activate
./scripts/download_models.sh
./scripts/verify_models.sh
```

`download-models` is interactive on a terminal: it shows model groups with
sizes and presence, then asks which groups to download. For scripts:

```bash
meeting-transcriber download-models --only all
meeting-transcriber download-models --only asr
meeting-transcriber download-models --only diarization
meeting-transcriber download-models --only llm
```

All model downloads are explicit host-side actions; inference never downloads
weights.

## How it works

```text
audio/video
   │  FFmpeg: extract track → 16 kHz mono WAV → 15-min chunks (15 s overlap)
   ▼
(optional) screen frames → dHash dedupe → Tesseract OCR → code cards C<n>
   ▼
faster-whisper (local CTranslate2 model)  →  raw segments + word timestamps
   ▼
Silero VAD → NeMo TitaNet-Large → spectral clustering  →  speaker intervals
   ▼
temporal alignment  →  segments with speaker_id + quality flags
   ▼
terminology glossary  →  auditable corrections (raw text preserved)
   ▼
optional llama.cpp report  →  topics, decisions, tasks, risks, questions
   ▼
JSON + Markdown + SRT (+ visual cards/frames, report.json / report.md)
```

ASR and diarization are loaded sequentially and released before the next stage.
The LLM runs only after both have released the GPU.

## Models

| Stage | Model | Revision | License |
|---|---|---|---|
| ASR | `deepdml/faster-whisper-large-v3-turbo-ct2` | `4df90f7` | MIT |
| VAD | Silero VAD 6.2.3 (bundled wheel) | — | MIT |
| Embeddings | `nvidia/speakerverification_en_titanet_large` | `0dc382f` | CC-BY-4.0 |
| Report LLM | `unsloth/Qwen3.5-9B-GGUF` (UD-Q4_K_XL) | `3885219` | Apache-2.0 |
| Video OCR | Tesseract 5 + `rus`/`eng` traineddata | host package | Apache-2.0 |

Details, sizes and rejected alternatives:
[`models/README.md`](models/README.md) and
[`docs/model-selection.md`](docs/model-selection.md).

## Outputs

By default, outputs are written next to the input file. With `--output-dir DIR`
they go to `DIR`.

```text
<dir>/meeting.json             # canonical contract: raw + corrected + audit
<dir>/meeting.markdown         # readable transcript (+ report if generated)
<dir>/meeting.srt              # subtitles with [speaker_XX]
<dir>/meeting.report.json      # only with the LLM report
<dir>/meeting.report.md
<dir>/meeting.visual.json      # only with video analysis: OCR cards and timestamps
<dir>/meeting.frames/          # only with video analysis: code-card frames
```

The JSON keeps `raw_text`, `corrected_text`, `speaker_id`, word timestamps,
terminology `corrections`, quality flags, model revisions, config snapshot and
run metrics. Existing outputs are never overwritten without `--force`; a
timestamp suffix is used instead.

## Common workflows

**Transcript only, no video, no report:**

```bash
meeting-transcriber transcribe -i meeting.mp4
```

**Video with code context, no report:**

```bash
meeting-transcriber transcribe-video -i meeting.mp4
```

**Transcript + report in one run:**

```bash
./scripts/start_llama_server.sh
meeting-transcriber transcribe -i meeting.mp4 --visual --llm
./scripts/stop_llama_server.sh
```

**Report later from an existing transcript:**

```bash
./scripts/start_llama_server.sh
meeting-transcriber report meeting.json
./scripts/stop_llama_server.sh
```

**Speaker names** (manual, never guessed):

```bash
meeting-transcriber speakers meeting.json
# edit meeting.speakers.yaml
meeting-transcriber label-speakers meeting.json --map meeting.speakers.yaml
```

**Multi-track video:**

```bash
ffprobe -hide_banner -show_streams meeting.mp4
meeting-transcriber transcribe -i meeting.mp4 --audio-stream 1
```

**Recovery** — if outputs are lost but `work/` exists, rebuild them without
re-running ASR:

```bash
meeting-transcriber transcribe -i meeting.mp4
```

## GPU and VRAM

| Process | VRAM |
|---|---|
| ASR + diarization | ~4.6 GiB |
| llama-server (Qwen3.5-9B, 8k context) | ~6.7 GiB |

On an 8 GB GPU, **never run both at once**:

1. stop `llama-server` → transcribe;
2. start `llama-server` → generate the report;
3. stop it before the next transcription.

```bash
./scripts/stop_llama_server.sh
curl -s http://127.0.0.1:8080/health || echo "server stopped"
```

Video OCR runs on the CPU and does not consume VRAM.

More: [`docs/gpu-memory.md`](docs/gpu-memory.md).

## Configuration

`config/default.yaml` is fully commented and validated; unknown keys are
rejected. Relative paths resolve against `paths.root`. Environment substitution
uses `{env:NAME}`.

The two settings that control the main modes are `visual.enabled` and
`llm.enabled`:

```yaml
asr:
  device: cuda
  compute_type: float16
  chunk_seconds: 900
  overlap_seconds: 15

diarization:
  max_speakers: 20
  num_speakers: null       # set to force an exact count

terminology:
  profiles: [general-it]
  threshold: 0.86
  apply_corrections: true

visual:
  enabled: false           # video OCR; override with --visual
  backend: tesseract
  languages: [rus, eng]
  frame_interval_seconds: 4
  crop: [0.22, 0.09, 0.90, 0.95]   # left, top, right, bottom
  keep_code_frames: true
  report_context_chars: 1200

llm:
  enabled: true            # used by the separate `report` command
  base_url: http://127.0.0.1:8080/v1
  model: qwen3.5-9b
  max_topics: 6
  max_items_per_section: 6
  report_style: asd-ste100   # "plain" = full report, no short-form caps
  chunk_chars: 5000
```

`transcribe` and `transcribe-video` skip the LLM report regardless of
`llm.enabled`; use `--llm` or the `report` command to generate it.

`report_style: asd-ste100` returns the short form (6 topics, 6 items per section,
5-sentence summary). `report_style: plain` returns the full report (12 topics,
no per-section cap, summary up to 10 sentences). Override per run with
`--report-style plain`.

## Docker

```bash
./scripts/build_image.sh                # ~8.14 GB, 15–40 min
./scripts/download_models.sh            # on the host

./scripts/compose.sh run --rm transcriber transcribe-video \
  -i /data/input/meeting.mp4 \
  --config /app/config/docker.yaml \
  --output-dir /data/output
```

Use `scripts/compose.sh`, not plain `docker compose`: it exports the host UID/GID
so bind-mounted `work/` and `data/output/` are writable. The container's
`/data/input` mount is read-only, so pass `--output-dir /data/output` inside
Docker. Models are mounted read-only; the image contains no weights. Compose
uses `network_mode: "none"`, so the LLM report runs natively on the host.

Build, verification, offline transfer and troubleshooting:
[`docs/docker.md`](docs/docker.md).

## Offline guarantees

- Missing weights exit with code 3 and a hint; the runtime never downloads them.
- `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set in backends, image,
  Compose and smoke tests.
- Compose runs with `network_mode: "none"`.
- Tesseract language data can be kept locally and pointed to by
  `visual.tessdata_dir`.
- `scripts/offline_smoke_test.sh` uses a private network namespace when
  available.

Details: [`docs/offline-deployment.md`](docs/offline-deployment.md).

## Testing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
ruff check src tests
```

Unit and integration tests use fake backends and need no GPU, FFmpeg or weights.
Real-model smoke test:

```bash
./scripts/offline_smoke_test.sh data/input/smoke.wav
```

## Documentation

- [Getting started](docs/getting-started.md)
- [CLI reference](docs/cli.md)
- [Docker](docs/docker.md)
- [Architecture and ADRs](docs/architecture.md)
- [Model selection](docs/model-selection.md)
- [Offline deployment](docs/offline-deployment.md)
- [GPU memory](docs/gpu-memory.md)
- [Quality evaluation](docs/quality-evaluation.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Known limitations](docs/known-limitations.md)

## Status

- 68 unit/integration tests; `ruff` clean.
- Native 87-minute run: 2m31s, 945 segments, 8 speakers, 4612 MiB peak VRAM.
- Docker image builds (8.14 GB) and runs the full recording: 2m40s, 918
  segments, 8 speakers, 4612 MiB peak.
- Native 48-minute video with visual analysis: 546 segments, 1 speaker,
  119 screen cards (109 with code).
- LLM report on the 65-minute concatenated meeting: 6 topics, 6 decisions,
  6 action items, 6 risks, 6 open questions (all capped).
- Report wording follows ASD-STE100-style rules by default.
- WER/DER against ground truth were not measured; the Docker LLM path is
  intentionally host-side; the base image tag is not digest-pinned.

## License

Code: MIT. Model weights retain their upstream licenses (MIT, CC-BY-4.0,
Apache-2.0).
