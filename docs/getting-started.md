# Getting started

This is the detailed, step-by-step guide. For a short overview see the
[README](../README.md).

## 0. Requirements

- Linux (tested on Ubuntu), NVIDIA GPU with **8 GB VRAM or more**;
- Python **3.10+**, FFmpeg and FFprobe;
- ~**12 GB** free disk: dependencies (~2–3 GB), models (~7.2 GB), temporary
  audio and chunks;
- a meeting recording: video (MP4/MKV/MOV/WebM/AVI/TS) or audio.

Check the host:

```bash
python3 --version
ffmpeg -version | head -n1
nvidia-smi
```

Repository layout:

```text
meeting-transcriber/
├── config/                 # default/docker configs, terminology profiles
├── data/input, data/output # source files and results (gitignored)
├── docker/                 # Dockerfile + entrypoint
├── docs/                   # guides, architecture, troubleshooting
├── models/                 # manifest + README (weights gitignored)
├── requirements/           # base / asr / diarization / dev
├── scripts/                # install, download, verify, build, llama.cpp
├── src/meeting_transcriber/
│   ├── audio/ asr/ diarization/ alignment/
│   ├── terminology/ postprocessing/ pipeline/ output/
│   └── cli.py config.py models.py types.py
├── tests/                  # unit + integration (no GPU needed)
├── compose.yaml
└── pyproject.toml
```

## 1. Install the environment

```bash
cd /home/demakov/meeting-transcriber

# FFmpeg + venv + PyTorch CUDA + NeMo + faster-whisper
./scripts/install_host_dependencies.sh

source .venv/bin/activate
meeting-transcriber --version
```

If `sudo` is unavailable, install FFmpeg manually and run the same script with
`--no-system`.

## 2. Download the models

Models are downloaded **only explicitly** and never during inference.

```bash
# everything: ASR + diarization + report LLM (~7.2 GiB)
./scripts/download_models.sh

# or selectively
./scripts/download_models.sh --only asr
./scripts/download_models.sh --only diarization
./scripts/download_models.sh --only llm

./scripts/verify_models.sh
meeting-transcriber list-models
```

| Group | Model | Size |
|---|---|---|
| `asr` | faster-whisper large-v3-turbo CT2 | ~1.5 GiB |
| `diarization` | TitaNet-Large `.nemo` | ~97 MiB |
| `llm` | unsloth Qwen3.5-9B UD-Q4_K_XL GGUF | ~5.6 GiB |

## 3. Install llama.cpp (report stage only)

```bash
./scripts/install_llama_cpp.sh
```

The script installs the pinned `b12032` CUDA build into
`~/.local/llama.cpp/b12032` (~600 MB), without sudo, and prints the
`llama-server` version.

## 4. VRAM rule

On an 8 GB GPU, ASR and the LLM must never run at the same time:

| Process | VRAM |
|---|---|
| ASR + diarization | ~4.6 GiB |
| `llama-server` (Qwen3.5-9B, 8k context) | ~6.7 GiB |

Order of operations:

1. `llama-server` stopped → run transcription;
2. transcription finished → start `llama-server` → generate the report;
3. stop `llama-server` before the next transcription.

Check or stop the server:

```bash
curl -s http://127.0.0.1:8080/health || echo "server stopped"
./scripts/stop_llama_server.sh
```

## 5. Transcribe and diarize

```bash
mkdir -p data/input
cp "/path/to/recording.webm" data/input/meeting.webm

meeting-transcriber transcribe data/input/meeting.webm \
  --config config/default.yaml \
  --skip-llm \
  --output-dir data/output
```

`--skip-llm` keeps the ASR/diarization stage separate from the LLM report, so
the GPU is not shared. The report is generated later with `report`.

Useful options:

```bash
# exact number of participants
--num-speakers 12

# upper bound for automatic speaker counting (default 20)
--max-speakers 20

# select an audio track in a multi-track video
--audio-stream 1

# recompute all stages and overwrite outputs
--force
```

What happens:

1. FFmpeg extracts the selected track to `work/<meeting>/audio_original.mka`.
2. A 16 kHz mono WAV is created and split into 15-minute chunks with 15 s
   overlap.
3. faster-whisper transcribes the speech.
4. NeMo diarization identifies speakers (up to 20).
5. Text and speakers are aligned and the terminology glossary is applied.

Reference: 87-minute recording on an RTX 4060 — **~2.5 minutes**, peak VRAM
**~4.6 GiB**.

Outputs:

```text
data/output/meeting.json       # raw + corrected text + audit trail
data/output/meeting.markdown   # readable transcript
data/output/meeting.srt        # subtitles with [speaker_XX]
```

Monitor progress and VRAM in another terminal:

```bash
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 2
```

## 6. Inspect the transcript

```bash
python - <<'PY'
import json
d = json.load(open("data/output/meeting.json"))
print("segments:", len(d["segments"]))
print("speakers:", len(d["speakers"]))
print("duration:", d["source_audio"]["duration"])
print("peak VRAM:", d["run"].get("vram_peak_mb"))
print("warnings:", d["warnings"])
PY

less data/output/meeting.markdown
head -40 data/output/meeting.srt
```

Check that:

- every segment has a `speaker_id` or an explicit `quality_flags` entry;
- no text is lost at chunk boundaries (around 15, 30, 45 minutes);
- technical terms and English insertions are correct;
- `corrections` contains only meaningful replacements.

## 7. Speaker names

Names are never guessed automatically.

```bash
meeting-transcriber speakers data/output/meeting.json
```

The command prints talk time, segment count and three sample quotes per speaker,
and creates:

```text
data/output/meeting.speakers.yaml
```

Fill the `name` fields:

```yaml
speakers:
  speaker_00:
    name: Ivan
  speaker_04:
    name: Maria
```

Apply the names without re-running ASR:

```bash
meeting-transcriber label-speakers data/output/meeting.json \
  --map data/output/meeting.speakers.yaml
```

Markdown shows `Ivan (speaker_00)`, SRT shows `[Ivan]`, JSON stores
`display_name`; `speaker_id` remains for auditability.

## 8. Meeting report with llama.cpp

```bash
./scripts/start_llama_server.sh

meeting-transcriber report data/output/meeting.json \
  --map data/output/meeting.speakers.yaml \
  --config config/default.yaml

./scripts/stop_llama_server.sh
```

If names are not filled yet, omit `--map`.

Outputs:

```text
data/output/meeting.report.json   # structured report with S<n> references
data/output/meeting.report.md     # standalone Markdown protocol
data/output/meeting.markdown      # transcript + appended protocol
```

Reference timings: first pass over an 87-minute recording — **~8–10 minutes**
(20 chunks + topic consolidation + summary). A rerun that reuses cached chunk
JSON takes **~2–3 minutes**. Always stop the server with
`./scripts/stop_llama_server.sh`.

## 9. Reruns, cache and recovery

- `work/<meeting>/` stores stage artifacts and `state.json`.
- A repeated `transcribe` without `--force` reuses completed stages.
- A configuration change invalidates only the affected stages (for example, a
  glossary edit does not re-run ASR).
- `--force` recomputes everything and overwrites outputs.
- Without `--force`, existing output files are not overwritten; a copy with a
  timestamp suffix is created.

If `data/output/` was deleted but `work/` is intact, rebuild in seconds:

```bash
meeting-transcriber transcribe data/input/meeting.webm \
  --config config/default.yaml --skip-llm

./scripts/start_llama_server.sh
meeting-transcriber report data/output/meeting.json --config config/default.yaml
./scripts/stop_llama_server.sh
```

A full recompute is needed only if `work/<meeting>/` is removed or the source
file changed.

## 10. Video and multiple audio tracks

```bash
# list tracks
ffprobe -hide_banner -show_streams data/input/meeting.mp4

# extract a track separately (optional)
meeting-transcriber extract-audio data/input/meeting.mp4 \
  --output data/output/meeting.mka --audio-stream 1 --force

# transcribe the selected track
meeting-transcriber transcribe data/input/meeting.mp4 \
  --config config/default.yaml --skip-llm --audio-stream 1
```

FFmpeg copies the selected track losslessly to `audio_original.mka` (FLAC
fallback if needed), then creates the 16 kHz mono WAV.

## 11. Quality knobs

| Setting | Where | Effect |
|---|---|---|
| `--num-speakers N` | CLI | exact speaker count instead of auto-detection |
| `--max-speakers N` | CLI / YAML | upper bound, default 20 |
| `terminology.profiles` | YAML | project-specific glossary |
| `terminology.threshold` | YAML | higher = fewer corrections |
| `llm.max_topics` | YAML | number of consolidated themes (default 6) |
| `llm.max_items_per_section` | YAML | max decisions/tasks/risks/questions (default 6) |
| `llm.report_style` | YAML | `asd-ste100` (default) or `plain` |
| `llm.chunk_chars` | YAML | LLM chunk size; smaller is safer for context |
| `LLAMA_CTX` | env at server start | llama-server context (default 8192) |

If the report is too granular, lower `max_topics`; if it lacks detail, raise it.
If terms are not corrected, add them to `config/terminology/*.yaml` and rerun
the terminology stage (ASR is reused without `--force`).

## 12. Docker

See [docker.md](docker.md). In short: ASR + diarization run in the container,
the LLM report runs natively on the host because Compose uses
`network_mode: "none"`.

## 13. Troubleshooting

Quick reference:

| Symptom | Action |
|---|---|
| `command not found: meeting-transcriber` | `source .venv/bin/activate` |
| `ffmpeg not found` | `./scripts/install_host_dependencies.sh` |
| `ASR model directory not found` | `./scripts/download_models.sh --only asr` |
| `speaker embedding checkpoint not found` | `./scripts/download_models.sh --only diarization` |
| CUDA OOM | stop `llama-server`; reduce `batch_size`/`beam_size` |
| `llama-server` not responding | `./scripts/start_llama_server.sh`, then `tail logs/llama-server.log` |
| empty report / `reasoning_content` | server must run with `--reasoning off` (the script does this) |
| consolidation context overflow | lower `llm.merge_chars` or raise `LLAMA_CTX` |
| `Speaker_00` in report text | regenerate the report with the current code |

Full list: [troubleshooting.md](troubleshooting.md).

## 14. Limitations

- Diarization does not separate overlapping speech; it only reports overlap.
- `speaker_XX` is not a person's name; names are assigned manually.
- WER/DER have not been measured against ground truth.
- The LLM report is a draft: verify `S<n>` references and uncertain items.

More: [known-limitations.md](known-limitations.md).
