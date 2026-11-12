# Offline deployment

## Principle

The network is needed exactly once: to download model weights on the host.
After that, inference runs without internet and without HF tokens.

## Build an offline bundle (connected machine)

```bash
# 1. Host dependencies
./scripts/install_host_dependencies.sh

# 2. Weights (only explicit network step)
./scripts/download_models.sh

# 3. Integrity check
./scripts/verify_models.sh

# 4. Docker image
./scripts/build_image.sh

# 5. Optional: prepare a tarball for transfer
docker save meeting-transcriber:0.1.0 | gzip > meeting-transcriber-image.tar.gz
tar czf meeting-transcriber-models.tar.gz models/
```

Dependencies inside the image are installed at build time; the image must be
built while the build host has access to PyPI. It never downloads weights.

## Deploy on the offline target

```bash
gunzip -c meeting-transcriber-image.tar.gz | docker load
tar xzf meeting-transcriber-models.tar.gz

./scripts/compose.sh run --rm transcriber \
  transcribe -i /data/input/meeting.webm \
  --config /app/config/docker.yaml \
  --output-dir /data/output
```

`./scripts/compose.sh` exports the host UID/GID so the bind-mounted `work/` and
`data/output/` are writable. `compose.yaml` already uses
`network_mode: "none"`. No ports are published. Models are mounted `:ro` at
`/models`.

## Native (non-Docker) offline run

```bash
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
meeting-transcriber transcribe -i data/input/meeting.wav --config config/default.yaml
```

## Local LLM report

`llm.enabled: true` targets llama.cpp `llama-server` at
`http://127.0.0.1:8080/v1` with unsloth Qwen3.5-9B UD-Q4_K_XL. The report runs
after ASR and diarization, so the GPU is free. If the server is unreachable the
transcript still completes; the pipeline only adds a warning.

Start and stop it with the provided scripts:

```bash
./scripts/install_llama_cpp.sh
./scripts/download_models.sh --only llm
./scripts/start_llama_server.sh
# ... generate a report ...
./scripts/stop_llama_server.sh   # free VRAM before the next ASR run
```

`docker compose` runs with `network_mode: "none"`, so the LLM is disabled in
`config/docker.yaml`. To use it from a container, switch to a bridge network and
point `llm.base_url` at the host.

The server holds ~6.7 GiB with an 8k context, so stop it before transcribing
another meeting on an 8 GB GPU.

## Verifying isolation

```bash
# Models are present and hashed
./scripts/verify_models.sh

# Pipeline works with no network namespace
unshare -rn ./scripts/offline_smoke_test.sh
```

`scripts/offline_smoke_test.sh` verifies weights, generates a short silence
fixture if none is supplied, and runs the pipeline in a private network
namespace when `unshare` is permitted.

## Failure mode when weights are missing

The pipeline does not fall back to the Hub. It exits with code `3` and a message
pointing at `./scripts/download_models.sh`:

```text
error: ASR model directory not found: /app/models/whisper-large-v3-turbo-ct2.
Run ./scripts/download_models.sh (runtime never downloads weights).
```

## Secrets

No Hugging Face token is required or accepted at runtime. `.env.example`
documents only optional local-LLM settings. Do not place tokens in `.env`,
Compose, or the image.

## Updates

Model updates are manual by design:

1. change `repo`/`revision`/URLs/SHA-256 in `models/manifest.yaml`;
2. `./scripts/download_models.sh --force --only <group>`;
3. `./scripts/verify_models.sh`;
4. rebuild or restart the deployment.
