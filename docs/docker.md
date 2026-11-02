# Docker

The image contains the application and Python dependencies, but **no model
weights**. Weights are downloaded on the host and mounted read-only.

## Requirements

- Docker Engine 24+ with Compose v2;
- NVIDIA driver and `nvidia-container-toolkit`;
- models downloaded on the host (`models/`, ~7.2 GiB);
- free disk: image ~8.14 GB plus build cache and models — 25–30 GB recommended;
- network access **during the build** (Docker Hub, PyPI); runtime is offline.

Verify GPU access for Docker:

```bash
docker info | grep -i nvidia
docker run --rm --gpus all pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime \
  python -c "import torch; print(torch.cuda.is_available())"
```

## Build

```bash
cd /home/demakov/meeting-transcriber
./scripts/build_image.sh
```

The script:

1. creates `work/`, `data/output/`, `logs/`;
2. builds `docker/Dockerfile`;
3. tags the image `meeting-transcriber:0.1.0`.

Custom tag:

```bash
IMAGE=meeting-transcriber:2026-11 ./scripts/build_image.sh
```

Clean rebuild:

```bash
./scripts/build_image.sh --no-cache
```

The Dockerfile uses `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`, installs
FFmpeg and the pinned requirements, installs the package, and switches to a
non-root user. `.dockerignore` excludes `.venv*`, `models/`, `data/`, `work/`,
`tests/`, and `docs/`, so the build context is small.

Measured image size: **8.14 GB**. Build time: 15–40 minutes depending on network
and cache.

## Verify the image

```bash
docker images meeting-transcriber:0.1.0

./scripts/compose.sh run --rm transcriber --help
./scripts/compose.sh run --rm transcriber validate-config \
  --config /app/config/docker.yaml
./scripts/compose.sh run --rm transcriber list-models \
  --config /app/config/docker.yaml
./scripts/compose.sh run --rm transcriber verify-models \
  --config /app/config/docker.yaml
./scripts/compose.sh run --rm transcriber python -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`scripts/compose.sh` is required for bind mounts. It exports `HOST_UID` and
`HOST_GID`, creates `work/` and `data/output/`, and runs `docker compose`. Plain
`docker compose run` uses UID 1000 from the image; on a host with another UID it
will fail with `permission denied` when writing to `work/` or `data/output/`.

## Run transcription

```bash
cp "/path/to/recording.webm" data/input/meeting.webm

./scripts/compose.sh run --rm transcriber transcribe \
  /data/input/meeting.webm \
  --config /app/config/docker.yaml \
  --skip-llm
```

Useful options:

```bash
--num-speakers 12
--max-speakers 20
--audio-stream 1
--force
```

`--skip-llm` is recommended: the LLM report is disabled in the container and
ASR must not compete with `llama-server` for VRAM. A containerized run of the
87-minute reference recording took **~2m40s**, found 8 speakers and peaked at
**4612 MiB**.

Outputs appear on the host:

```text
data/output/meeting.json
data/output/meeting.markdown
data/output/meeting.srt
```

If files already exist, a timestamp suffix is added unless `--force` is used.
`work/<meeting>/` stores stage artifacts for resume.

## Volumes

| Host | Container | Mode | Purpose |
|---|---|---|---|
| `./models` | `/models` | `ro` | weights + `manifest.yaml` / lock |
| `./data/input` | `/data/input` | `ro` | source audio/video |
| `./data/output` | `/data/output` | `rw` | JSON/Markdown/SRT/report |
| `./work` | `/work` | `rw` | chunks, stages, resume |
| `./config` | `/app/config` | `ro` | configs and glossaries |

The image never downloads weights. `verify-models` reads
`/models/manifest.yaml`.

## Report

The LLM is disabled in the container because `compose.yaml` uses
`network_mode: "none"` and the model needs ~6.7 GiB VRAM. Generate the report
natively from the mounted outputs:

```bash
./scripts/start_llama_server.sh
meeting-transcriber report data/output/meeting.json \
  --map data/output/meeting.speakers.yaml \
  --config config/default.yaml
./scripts/stop_llama_server.sh
```

To run the LLM inside a container, switch Compose to a bridge network, start
`llama-server` on the host, and set `llm.enabled: true` with
`llm.base_url: http://host.docker.internal:8080/v1` in
`config/docker.yaml`. On an 8 GB GPU, ASR and the LLM still must not run at the
same time.

## Offline transfer

On a connected machine:

```bash
docker save meeting-transcriber:0.1.0 | gzip > meeting-transcriber-image.tar.gz
tar czf meeting-transcriber-models.tar.gz models/
```

On the offline target:

```bash
gunzip -c meeting-transcriber-image.tar.gz | docker load
tar xzf meeting-transcriber-models.tar.gz

cd /home/demakov/meeting-transcriber
./scripts/compose.sh run --rm transcriber transcribe \
  /data/input/meeting.webm \
  --config /app/config/docker.yaml \
  --skip-llm
```

Without the repository, run the image directly:

```bash
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/models:/models:ro" \
  -v "$PWD/data/input:/data/input:ro" \
  -v "$PWD/data/output:/data/output" \
  -v "$PWD/work:/work" \
  -v "$PWD/config:/app/config:ro" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e HOME=/tmp -e NUMBA_CACHE_DIR=/tmp/numba -e MPLCONFIGDIR=/tmp/matplotlib \
  meeting-transcriber:0.1.0 transcribe /data/input/meeting.webm \
  --config /app/config/docker.yaml --skip-llm
```

## Update and clean up

After code or dependency changes:

```bash
./scripts/build_image.sh
```

Models and outputs are on the host and are not touched. Remove the old image and
build cache:

```bash
docker image rm meeting-transcriber:0.1.0
docker builder prune
docker system df
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `permission denied` in `work/` or `data/output/` | use `./scripts/compose.sh`, not plain `docker compose` |
| `could not select device driver ... gpu` | `nvidia-container-toolkit` missing |
| `CUDA out of memory` | `llama-server` is still running; stop it |
| `cannot cache function '__o_fold'` | numba cache not writable; use `compose.sh` (sets env) |
| `model not found` / empty `/models` | `models/` not mounted or `models/manifest.yaml` missing |
| `network_mode: "none"` and `report` | expected; run the report natively |
| build fails at `pip install` | check PyPI access and free disk |
