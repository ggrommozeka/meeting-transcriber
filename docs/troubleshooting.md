# Troubleshooting

## `required executable 'ffmpeg' was not found`

```bash
sudo apt-get install -y ffmpeg
```

Or run `./scripts/install_host_dependencies.sh`.

## Video file has no usable audio

`ffprobe` fails with `contains no audio stream (video without an audio track?)`.
Check the file:

```bash
ffprobe -hide_banner -show_streams meeting.mp4
```

If the video really has no audio track, transcription is impossible. If it has
several tracks, select one with `--audio-stream N` (0-based, FFmpeg `0:a:N`).

## `requested stream N` for a video

The file has fewer audio tracks than requested. The error lists the valid range.
Run `ffprobe` or `meeting-transcriber extract-audio` to inspect the tracks.

## `ASR model directory not found` / `speaker embedding checkpoint not found`

Weights are not present. The runtime never downloads them.

```bash
./scripts/download_models.sh
./scripts/verify_models.sh
```

## `faster-whisper is not installed`

```bash
pip install -r requirements/asr.txt
```

## `NeMo ASR toolkit is not installed`

```bash
pip install -r requirements/diarization.txt
```

NeMo 2.6.2 requires `torch>=2.6`. On a fresh host install a CUDA build first:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

## `unexpected keyword argument 'metadata_errors'`
faster-whisper 1.1.1 calls `av.open(metadata_errors=...)`, which PyAV removed in
17.0. `requirements/asr.txt` pins `av>=11,<17`. Fix an existing environment:

```bash
pip install "av>=11,<17"
```

## `Library libcublas.so.12 is not found`

The CLI now re-execs itself with pip-installed `nvidia-*` libraries added to
`LD_LIBRARY_PATH`, so normally nothing needs to be exported. If the libraries
are missing entirely:

```bash
pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*" nvidia-cuda-runtime-cu12
```

For custom code that imports `WhisperBackend` directly, export them manually:

```bash
export LD_LIBRARY_PATH="$(find .venv/lib/python3.13/site-packages/nvidia -maxdepth 2 -type d -name lib | paste -sd:):$LD_LIBRARY_PATH"
```

## CUDA errors / `no kernel image is available`

The driver, PyTorch CUDA build, and CTranslate2 build must match. Check:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import ctranslate2; print(ctranslate2.__version__, ctranslate2.get_cuda_device_count())"
```

Fallback for testing: `--device cpu`.

## Diarization failed but transcription succeeded

By default (`pipeline.continue_on_diarization_error: true`) the pipeline keeps
going and records the error in `warnings`. Set the flag to `false` to fail hard.

## All segments have `speaker_id: null`

Look at `quality_flags`:

* `no_diarization` — diarization disabled or failed;
* `no_speaker_match` — no diarization interval covers the segment;
* `ambiguous_speaker_assignment` — two speakers explain a similar share of the
  segment; the best label is kept but flagged.

Also check `diarization_intervals` in the JSON.

## Too many / too few speakers

* Provide the exact count: `--num-speakers 12`.
* Raise the upper bound: `--max-speakers 20`.
* Short utterances (< 1 s) are inherently unstable for embeddings.

NeMo's automatic counting is not perfect; documenting and using the manual count
is expected.

## Terminology over-corrects

* Raise `terminology.threshold` (e.g. `0.92`).
* Disable fuzzy matching: `terminology.fuzzy: false`.
* Keep suggestions without applying:
  `terminology.apply_corrections: false`; corrections are still recorded.
* Use project-specific profiles instead of broad ones.

## LLM report is missing

* `llm.enabled` must be `true` (default config) and `llama-server` must be
  running: `./scripts/start_llama_server.sh`.
* Check the endpoint: `curl http://127.0.0.1:8080/v1/models`.
* Check the log: `tail -n 50 logs/llama-server.log`.
* If the server was installed in a custom location, set `LLAMA_SERVER`.
* `scripts/start_llama_server.sh` fails if `./scripts/install_llama_cpp.sh` or
  `./scripts/download_models.sh --only llm` was not run.
* The report is always optional; transcript outputs are still written.
* `--skip-llm` disables it for one run.

## `meeting report: chunk N skipped (invalid JSON after retries)`

The chunk did not fit the model context window. Lower `llm.chunk_chars`
(default 5000) and re-run `report`; cached successful chunks are reused.
Generated chunks live in `work/<meeting>/report_chunks/`.

## Report lost decisions/tasks

The reduce phase is deterministic and keeps distinct items. If all chunks except
a few failed JSON parsing, those parts are missing — lower `llm.chunk_chars` and
re-run. Check the warnings on stderr and `report_chunks/` for the failed parts.

## Report is missing / has no summary

If the model returned only fragments, check stderr warnings. The title and
summary are generated in a final LLM call; if that call fails, the report still
contains the extracted topics/decisions/tasks but may have a fallback title.
Re-run `report` after checking the server.

## `request exceeds the available context size`

Topic consolidation is split into context-sized groups, but chunk or summary
prompts can still overflow with a custom setup. Restart `llama-server` with a
larger context (`LLAMA_CTX=16384 ./scripts/start_llama_server.sh`) if VRAM
allows, or lower `llm.chunk_chars`.

## `network_mode: "none"` and the local LLM

A container with no network cannot reach a host LLM. Either:

* run the LLM on the same host and temporarily use a bridge network; or
* run the LLM step natively; or
* keep the LLM disabled in the offline container (`config/docker.yaml`).

The offline requirement applies to transcription, not to the optional LLM.

## Output file name got a timestamp suffix

Existing results are never overwritten by default. Pass `--force` to overwrite,
or delete the previous outputs.

## Resume uses stale results

Cached stages are tied to a configuration fingerprint. If something looks stale:

```bash
meeting-transcriber transcribe meeting.wav --force
```

or delete `work/<meeting_id>/`.
