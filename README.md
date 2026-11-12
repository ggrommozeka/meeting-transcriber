# meeting-transcriber

Fully local transcription of Russian/mixed-language meetings with speaker
diarization, terminology correction, optional screen OCR and an optional local
LLM report. No cloud services and no model downloads during inference.

## Quick start

Requirements: Linux, Python 3.10+, FFmpeg, NVIDIA GPU with 8 GB+ VRAM.
Video OCR additionally needs Tesseract with `rus` and `eng` language data.

```bash
./scripts/install_host_dependencies.sh
source .venv/bin/activate
./scripts/download_models.sh        # interactive: choose model groups
./scripts/verify_models.sh
```

## Interactive wizards

```bash
meeting-transcriber transcribe -i /path/to/meeting.mp4
```

With OCR of slides/code shown in the video:

```bash
meeting-transcriber transcribe-video -i /path/to/meeting.mp4
```

Useful options:

```bash
--output-dir DIR        # write outputs elsewhere
--language ru           # ASR language
--skip-diarization      # ASR only
--num-speakers N        # exact speaker count
--audio-stream N        # select a 0-based audio track
--visual / --no-visual  # enable/disable video OCR
--keep-frames           # save code frames
```

The same two operations from Python:

```python
from meeting_transcriber import transcribe_audio, transcribe_video

transcribe_audio("meeting.mp4")
transcribe_video("meeting.mp4", output_dir="out")
```

## Interactive wizards

```bash
meeting-transcriber setup      # prepare the machine
meeting-transcriber process    # convert a recording into a report
```

`setup` selects paths, OCR settings, llama.cpp source (`existing` binary,
prebuilt `release`, or `build` from the latest stable tag), model groups, then
shows a readiness table with per-component and overall percentages. Choices are
saved to `config/local.yaml`.

`process` selects input/output and processing options, validates only required
components, and runs the pipeline with live stage progress. Pass `-i` and
`--output-dir` to prefill it.

Both wizards need a terminal; for scripts use `transcribe` / `transcribe-video`.

## Video OCR

Video analysis is **off by default**. Enable it with `transcribe-video`,
`--visual`, or:

```yaml
visual:
  enabled: true
```

It extracts stable screen frames, deduplicates them, runs local Tesseract OCR,
classifies code/slides and stores:

```text
meeting.visual.json
meeting.frames/
```

When a report is generated, report items link to code cards as
`код: C<n> (00:12:22)` and Markdown gets a “Код и материалы на экране” appendix.

## LLM report

The report uses a local `llama-server`. The tool starts it when needed and
stops it immediately after the report finishes:

```bash
./scripts/install_llama_cpp.sh          # once
meeting-transcriber report /path/to/meeting.json
```

Or generate it in the same run:

```bash
meeting-transcriber transcribe -i meeting.mp4 --visual --llm
```

Both paths stop the server automatically when generation is complete. If the
server was started manually, its PID file is used for cleanup; a server without
a PID file is left untouched.

On an 8 GB GPU, run transcription and the LLM server separately: ASR (~4.6 GiB)
and llama-server (~6.7 GiB) must not compete for VRAM. Video OCR runs on CPU.

Report modes:

```bash
--report-mode task_discussion   # 1:1 task discussion, C++/architecture focus
--report-mode status_meeting    # progress updates and new tasks
--prompt "..."                  # custom instructions for one run
--prompt-file prompt.md
```

In `meeting-transcriber process` the mode is selected on a separate screen
after enabling the LLM report; the custom prompt is entered there and is not
saved.

## Outputs

By default, files are written next to the input. With `--output-dir DIR` they go
to `DIR`.

```text
meeting.json             # canonical transcript + corrections + metadata
meeting.markdown         # readable transcript (+ report)
meeting.srt              # subtitles
meeting.report.json/.md   # only with the LLM report
meeting.visual.json      # only with video OCR
meeting.frames/          # only with video OCR: code frames
```

Existing outputs are not overwritten without `--force`; a timestamp suffix is
used instead.

## Configuration

`config/default.yaml` is validated and rejects unknown keys. The most important
switches are:

```yaml
asr:
  device: cuda
  compute_type: float16

diarization:
  max_speakers: 20
  num_speakers: null       # force an exact count

visual:
  enabled: false           # override with --visual
  backend: tesseract
  crop: [0.22, 0.09, 0.90, 0.95]

runtime:
  llama_server: null       # filled by `setup`
  llama_source: existing   # existing | release | build
  llama_tag: latest
  llama_build_dir: ~/.local/llama.cpp
  llama_cuda: true

llm:
  enabled: true            # used by the `report` command
  base_url: http://127.0.0.1:8080/v1
  model: qwen3.5-9b
  report_mode: general     # general | task_discussion | status_meeting
  custom_prompt: null      # one-run override; replaces report_mode
  report_style: asd-ste100 # or plain
```

Local machine-specific settings from `setup` live in `config/local.yaml`
(gitignored).

Report focus can be changed per run with `--report-mode task_discussion` or
`--report-mode status_meeting`, or with a one-off `--prompt` / `--prompt-file`.
The `process` wizard asks for this on a separate screen before starting.

## Docker

```bash
./scripts/build_image.sh
./scripts/download_models.sh

./scripts/compose.sh run --rm transcriber transcribe-video \
  -i /data/input/meeting.mp4 \
  --config /app/config/docker.yaml \
  --output-dir /data/output
```

Use `scripts/compose.sh` so bind-mounted `work/` and `data/output/` are
writable. The input mount is read-only, so always pass `--output-dir` inside
Docker. Compose runs with `network_mode: "none"`; generate reports on the host.

## Models

| Stage | Model |
|---|---|
| ASR | `deepdml/faster-whisper-large-v3-turbo-ct2` |
| VAD | Silero VAD 6.2.3 |
| Embeddings | `nvidia/speakerverification_en_titanet_large` |
| Report LLM | `unsloth/Qwen3.5-9B-GGUF` (UD-Q4_K_XL) |
| Video OCR | Tesseract 5 + `rus`/`eng` traineddata |

Weights are downloaded explicitly by `download_models` and verified by SHA-256;
runtime never downloads them.

## Troubleshooting

- **No GPU / CUDA error** — transcription is CPU-capable but slow; check
  `nvidia-smi` and the PyTorch CUDA build.
- **`tesseract` not found** — install `tesseract-ocr` with `rus`/`eng`, or run
  with `--no-visual`.
- **Report fails** — start `llama-server` first and check
  `curl http://127.0.0.1:8080/health`.
- **Missing models** — run `./scripts/download_models.sh --only all`.

## More documentation

- [CLI reference](docs/cli.md)
- [Getting started](docs/getting-started.md)
- [Docker](docs/docker.md)
- [Offline deployment](docs/offline-deployment.md)
- [GPU memory](docs/gpu-memory.md)
- [Troubleshooting](docs/troubleshooting.md)

## Testing

```bash
pip install -e ".[dev]"
pytest -q
ruff check src tests
```

77 unit/integration tests; `ruff` clean.

## License

Code: MIT. Model weights retain their upstream licenses.
