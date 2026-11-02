# meeting-transcriber

Fully local transcription of meetings with Russian speech and mixed
Russian/English technical terminology, speaker diarization for roughly 8–15
participants, terminology correction, and an optional local-LLM meeting report.

No cloud services, no Hugging Face token at runtime, and no model downloads
during inference.

## Key properties

- **Video or audio input** — FFmpeg extracts the selected audio track losslessly.
- **Local ASR** — faster-whisper / Whisper large-v3-turbo, CTranslate2.
- **Diarization without a 4-speaker cap** — Silero VAD → TitaNet-Large → NeMo
  spectral clustering, up to 20 speakers.
- **Auditable corrections** — raw ASR text is preserved; every terminology
  replacement is recorded.
- **Fully offline runtime** — weights are mounted local files; network is used
  only by explicit download scripts.
- **8 GB VRAM target** — ASR and diarization run sequentially on one GPU.

## Quick start

Requirements: Linux, Python 3.10+, FFmpeg, an NVIDIA GPU with 8 GB+ VRAM
(CPU works for testing only).

```bash
# 1. Host packages, venv, PyTorch CUDA, NeMo, faster-whisper
./scripts/install_host_dependencies.sh
source .venv/bin/activate

# 2. Models (~7.2 GiB; the only network step for weights)
./scripts/download_models.sh
./scripts/verify_models.sh

# 3. Transcribe and diarize (LLM report is a separate step)
meeting-transcriber transcribe data/input/meeting.webm \
  --config config/default.yaml \
  --skip-llm \
  --output-dir data/output
```

Then assign speaker names and generate the protocol:

```bash
meeting-transcriber speakers data/output/meeting.json
# fill in data/output/meeting.speakers.yaml

./scripts/install_llama_cpp.sh          # once
./scripts/start_llama_server.sh
meeting-transcriber report data/output/meeting.json \
  --map data/output/meeting.speakers.yaml \
  --config config/default.yaml
./scripts/stop_llama_server.sh
```

Full walkthrough, including recovery and multi-track video:
[`docs/getting-started.md`](docs/getting-started.md).

## How it works

```text
audio/video
   │  FFmpeg: extract track → 16 kHz mono WAV → 15-min chunks (15 s overlap)
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
JSON + Markdown + SRT (+ report.json / report.md)
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

Details, sizes and rejected alternatives:
[`models/README.md`](models/README.md) and
[`docs/model-selection.md`](docs/model-selection.md).

## Outputs

```text
data/output/meeting.json       # canonical contract: raw + corrected + audit
data/output/meeting.markdown   # readable transcript (+ report if generated)
data/output/meeting.srt        # subtitles with [speaker_XX]
data/output/meeting.report.json
data/output/meeting.report.md
```

The JSON keeps `raw_text`, `corrected_text`, `speaker_id`, word timestamps,
terminology `corrections`, quality flags, model revisions, config snapshot and
run metrics. Existing outputs are never overwritten without `--force`; a
timestamp suffix is used instead.

## Common workflows

**ASR only:**

```bash
meeting-transcriber transcribe data/input/meeting.webm --skip-diarization
```

**Speaker names** (manual, never guessed):

```bash
meeting-transcriber speakers data/output/meeting.json
# edit data/output/meeting.speakers.yaml
meeting-transcriber label-speakers data/output/meeting.json \
  --map data/output/meeting.speakers.yaml
```

**Meeting report** — requires the local llama.cpp server:

```bash
./scripts/start_llama_server.sh
meeting-transcriber report data/output/meeting.json --config config/default.yaml
./scripts/stop_llama_server.sh
```

**Multi-track video:**

```bash
ffprobe -hide_banner -show_streams data/input/meeting.mp4
meeting-transcriber transcribe data/input/meeting.mp4 --audio-stream 1
```

**Recovery** — if `data/output/` is lost but `work/` exists, rebuild outputs
without re-running ASR:

```bash
meeting-transcriber transcribe data/input/meeting.webm --skip-llm
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

More: [`docs/gpu-memory.md`](docs/gpu-memory.md).

## Configuration

`config/default.yaml` is fully commented and validated; unknown keys are
rejected. Relative paths resolve against `paths.root`. Environment substitution
uses `{env:NAME}`.

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

llm:
  enabled: true
  base_url: http://127.0.0.1:8080/v1
  model: qwen3.5-9b
  max_topics: 12
  chunk_chars: 5000
```

## Docker

```bash
./scripts/build_image.sh                # ~8.14 GB, 15–40 min
./scripts/download_models.sh            # on the host

./scripts/compose.sh run --rm transcriber transcribe \
  /data/input/meeting.webm \
  --config /app/config/docker.yaml \
  --skip-llm
```

Use `scripts/compose.sh`, not plain `docker compose`: it exports the host UID/GID
so bind-mounted `work/` and `data/output/` are writable. Models are mounted
read-only; the image contains no weights. Compose uses `network_mode: "none"`,
so the LLM report runs natively on the host.

Build, verification, offline transfer and troubleshooting:
[`docs/docker.md`](docs/docker.md).

## Offline guarantees

- Missing weights exit with code 3 and a hint; the runtime never downloads them.
- `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set in backends, image,
  Compose and smoke tests.
- Compose runs with `network_mode: "none"`.
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

- 61 unit/integration tests; `ruff` clean.
- Native 87-minute run: 2m31s, 945 segments, 8 speakers, 4612 MiB peak VRAM.
- Docker image builds (8.14 GB) and runs the full recording: 2m40s, 918
  segments, 8 speakers, 4612 MiB peak.
- LLM report on the same meeting: 12 topics, 6 decisions, 10 action items.
- WER/DER against ground truth were not measured; the Docker LLM path is
  intentionally host-side; the base image tag is not digest-pinned.

## License

Code: MIT. Model weights retain their upstream licenses (MIT, CC-BY-4.0,
Apache-2.0).
