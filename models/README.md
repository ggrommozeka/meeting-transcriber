# Local models

Model weights are **not** committed and **not** embedded in the Docker image.
They are downloaded explicitly on the host by `./scripts/download_models.sh`
and mounted into the container read-only.

## Layout

```text
models/
├── manifest.yaml                                  # pins repo, revision, license, SHA-256
├── manifest.lock.json                             # generated on first download
├── whisper-large-v3-turbo-ct2/
│   ├── model.bin
│   ├── config.json
│   ├── preprocessor_config.json
│   ├── tokenizer.json
│   ├── vocabulary.json
│   └── LICENSE.txt
├── speakerverification_en_titanet_large/
│   ├── speakerverification_en_titanet_large.nemo
│   └── LICENSE.txt
└── qwen3.5-9b-gguf/
    ├── Qwen3.5-9B-UD-Q4_K_XL.gguf
    └── LICENSE.txt
```

## Models

### ASR: faster-whisper-large-v3-turbo-ct2

| Field | Value |
| --- | --- |
| Model | `deepdml/faster-whisper-large-v3-turbo-ct2` |
| Source | <https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2> |
| Revision | `4df90f75321148c3a29a9e2351b7ddf8f5b115a8` |
| License | MIT |
| Format | CTranslate2 (int8/float16 weights, `model.bin`) |
| Size | ~1.51 GiB |
| Language | multilingual, including `ru` and `en` |
| Runtime | `faster-whisper` loads the directory directly with `local_files_only=True` |
| Verified GPU | intended for 8 GB VRAM with `compute_type=float16`, `beam_size=5` |
| Limits | turbo is a pruned large-v3 variant; long-form accuracy is slightly below large-v3 |

The upstream `openai/whisper-large-v3-turbo` checkpoint is MIT licensed. The
CTranslate2 conversion is a third-party artifact; pinning the revision and
verifying SHA-256 makes it reproducible. If your policy requires first-party
weights only, convert `openai/whisper-large-v3-turbo` yourself with
`ct2-transformers-converter` and point `asr.model_dir` at the result.

### Diarization: TitaNet-Large speaker embeddings

| Field | Value |
| --- | --- |
| Model | `nvidia/speakerverification_en_titanet_large` |
| Source | <https://huggingface.co/nvidia/speakerverification_en_titanet_large> |
| Revision | `0dc382f40121a5fbd34db10a2bb04d826c2be6a8` |
| License | CC-BY-4.0 |
| Format | NeMo `.nemo` archive |
| Size | ~97 MiB |
| Runtime | `ClusteringDiarizer` restores it from the local path |
| Limits | embedding model is trained on English data; it is widely used as a language-agnostic speaker encoder, but Russian-specific DER is not guaranteed |

### VAD: Silero VAD (bundled in a Python wheel)

| Field | Value |
| --- | --- |
| Package | `silero-vad==6.2.3` |
| License | MIT |
| Format | JIT model bundled with the wheel |
| Runtime | produces speech intervals; NeMo consumes them via `vad.external_vad_manifest` |

Using Silero avoids a separate NGC `vad_multilingual_marblenet` download and
keeps the offline bundle small. The plan allowed a multilingual MarbleNet VAD;
Silero was chosen because it is language-agnostic, permissively licensed, and
available without NGC authentication. This is recorded in
`docs/model-selection.md`.

### LLM report: Qwen3.5-9B UD-Q4_K_XL (unsloth)

| Field | Value |
| --- | --- |
| Model | `unsloth/Qwen3.5-9B-GGUF` → `Qwen3.5-9B-UD-Q4_K_XL.gguf` |
| Source | <https://huggingface.co/unsloth/Qwen3.5-9B-GGUF> |
| Revision | `3885219b6810b007914f3a7950a8d1b469d598a5` |
| License | Apache-2.0 |
| Format | GGUF, unsloth dynamic Q4_K_XL |
| Size | 5.56 GiB |
| Runtime | llama.cpp `llama-server` b12032, CUDA 13.4 build |
| Context | 8192 (set at server start; model supports 262k) |
| VRAM | ~6.7 GiB with 8k context on an RTX 4060 8 GB |
| Limits | thinking is disabled (`--reasoning off`); server must be stopped before ASR |

Install llama.cpp and the model:

```bash
./scripts/install_llama_cpp.sh
./scripts/download_models.sh --only llm
./scripts/start_llama_server.sh
```

The pinned llama.cpp build is `b12032` (CUDA 13.4 x64). It is installed under
`~/.local/llama.cpp/b12032` without sudo.

## First-time setup

```bash
./scripts/download_models.sh          # ASR + diarization + report LLM
./scripts/download_models.sh --only asr
./scripts/download_models.sh --only llm
./scripts/verify_models.sh
```

The downloader:

* pins every URL to a commit revision;
* checks file size and SHA-256 (known hashes for LFS files, lock file for the rest);
* skips already-correct files on re-run;
* writes `LICENSE.txt` next to each model;
* never runs as part of the inference pipeline.

If a source is unreachable, the command fails with a clear error. There is no
silent fallback and no runtime download.

## Custom location

```bash
./scripts/download_models.sh --models-dir /srv/models
./scripts/verify_models.sh --models-dir /srv/models
meeting-transcriber transcribe meeting.wav --models-dir /srv/models
```

## Licensing

The project code is MIT. Model weights keep their upstream licenses:
MIT for the Whisper CT2 conversion and CC-BY-4.0 for TitaNet-Large. Review
`LICENSE.txt` in each model directory before redistribution.
