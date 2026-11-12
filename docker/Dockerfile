FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/hf-home

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg curl ca-certificates \
        tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements/ /app/requirements/
COPY pyproject.toml /app/
COPY src/ /app/src/

RUN pip install --upgrade pip \
    && pip install -r /app/requirements/asr.txt \
    && pip install -r /app/requirements/diarization.txt \
    && pip install --no-deps -e /app

COPY config/ /app/config/
COPY scripts/ /app/scripts/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN useradd --create-home --uid 1000 --shell /bin/bash transcriber \
    && chmod +x /usr/local/bin/entrypoint.sh /app/scripts/*.sh \
    && mkdir -p /app/work /app/data/input /app/data/output \
    && chown -R transcriber:transcriber /app

USER transcriber

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--help"]
