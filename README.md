# meeting-transcriber

Fully local transcription of meetings with Russian speech and mixed
Russian/English technical terminology, speaker diarization for roughly 8–15
participants, terminology correction, and an optional local-LLM meeting report.

**No cloud services, no HF token at runtime, no model downloads during inference.**

## What it does

1. Accepts audio **or video** (MP4, MKV, MOV, WebM, AVI, …). FFmpeg extracts
   the selected audio track losslessly to `audio_original.mka`, then converts it
   to 16 kHz mono WAV and splits long files into overlapping chunks.
2. Transcribes with faster-whisper (Whisper large-v3-turbo, CTranslate2) using a
   local model directory, word timestamps, and an optional technical prompt.
3. Diarizes with a NeMo cascade: Silero VAD → TitaNet-Large embeddings →
   spectral clustering (automatic speaker count up to `max_speakers`, default 20).
4. Aligns text and speakers by temporal intersection; splits at speaker changes
   when word timestamps are available; flags ambiguous assignments.
5. Corrects technical terms from a versioned YAML glossary with exact and fuzzy
   matching (phonetic matching is opt-in). Every replacement is recorded; raw
   text is preserved.
6. Optionally generates meeting minutes through a local OpenAI-compatible LLM,
   validating every segment reference.
7. Writes canonical JSON, Markdown, and SRT.

## Пошаговый гайд (рекомендуемый порядок)

Ниже — полный порядок действий от чистой машины до готового протокола.
Команды выполняются из корня репозитория.

### 0. Что понадобится

- Linux (проверено на Ubuntu), NVIDIA GPU с **8 ГБ VRAM и больше**;
- Python **3.10+**, FFmpeg/FFprobe;
- ~**12 ГБ** свободного места: зависимости ~2–3 ГБ, модели ~7.2 ГБ, рабочее
  пространство под аудио и чанки;
- файл встречи: видео (MP4/MKV/MOV/WebM/AVI/TS) или аудио.

Проверка:

```bash
python3 --version
ffmpeg -version | head -n1
nvidia-smi
```

### 1. Установка окружения

```bash
cd /home/demakov/meeting-transcriber

# ffmpeg + venv + PyTorch CUDA + NeMo + faster-whisper
./scripts/install_host_dependencies.sh

source .venv/bin/activate
meeting-transcriber --version
```

Если `sudo` недоступен, установите FFmpeg отдельно, затем выполните тот же
скрипт с `--no-system`.

### 2. Загрузка моделей

Модели скачиваются **только явно** и никогда во время инференса.

```bash
# всё сразу: ASR + diarization + LLM (~7.2 ГиБ)
./scripts/download_models.sh

# или выборочно
./scripts/download_models.sh --only asr
./scripts/download_models.sh --only diarization
./scripts/download_models.sh --only llm

# проверка размеров и SHA-256
./scripts/verify_models.sh
meeting-transcriber list-models
```

Что скачивается:

| Группа | Модель | Размер |
|---|---|---|
| `asr` | faster-whisper large-v3-turbo CT2 | ~1.5 ГиБ |
| `diarization` | TitaNet-Large `.nemo` | ~97 МиБ |
| `llm` | unsloth Qwen3.5-9B UD-Q4_K_XL GGUF | ~5.6 ГиБ |

### 3. Установка llama.cpp (только для протокола)

```bash
./scripts/install_llama_cpp.sh
```

Скрипт без `sudo` ставит закреплённую сборку `b12032` с CUDA в
`~/.local/llama.cpp/b12032` (~600 МБ) и проверяет `llama-server --version`.

### 4. Главное правило VRAM

На GPU 8 ГБ **нельзя одновременно** держать ASR и LLM:

| Процесс | VRAM |
|---|---|
| ASR + diarization | ~4.6 ГиБ |
| llama-server (Qwen3.5-9B, 8k ctx) | ~6.7 ГиБ |

Поэтому порядок строго такой:

1. `llama-server` **остановлен** → запускается транскрипция;
2. транскрипция завершена → запускается `llama-server` → генерируется протокол;
3. `llama-server` **останавливается** перед следующей транскрипцией.

Проверить/остановить сервер:

```bash
curl -s http://127.0.0.1:8080/health || echo "server stopped"
./scripts/stop_llama_server.sh
```

### 5. Транскрипция и diarization

```bash
mkdir -p data/input
cp "/путь/к/записи.webm" data/input/meeting.webm

meeting-transcriber transcribe data/input/meeting.webm \
  --config config/default.yaml \
  --skip-llm \
  --output-dir data/output
```

Почему `--skip-llm`: протокол генерируется отдельным шагом, когда
`llama-server` уже запущен. Так ASR не конкурирует с LLM за VRAM.

Полезные опции:

```bash
# если известно точное число участников
--num-speakers 12

# верхняя граница автоматического подсчёта (по умолчанию 20)
--max-speakers 20

# выбрать аудиодорожку в многодорожечном видео
--audio-stream 1

# полный пересчёт всех стадий (игнорировать кэш)
--force
```

Что происходит:

1. FFmpeg вырезает выбранную дорожку в `work/<meeting>/audio_original.mka`;
2. создаётся `audio_16k_mono.wav` и чанки по 15 минут с перекрытием 15 с;
3. faster-whisper распознаёт речь;
4. NeMo diarization определяет говорящих (до 20);
5. текст и говорящие объединяются, применяется глоссарий.

Ориентиры для записи 87 минут на RTX 4060: **~2.5 минуты**, пик VRAM
**~4.6 ГиБ**.

Результаты:

```text
data/output/meeting.json       # полный контракт: raw + corrected + аудит
data/output/meeting.markdown   # читаемая расшифровка
data/output/meeting.srt        # субтитры с [speaker_XX]
```

Прогресс и VRAM можно смотреть в другом терминале:

```bash
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 2
```

### 6. Проверка транскрипта

```bash
# краткая сводка
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

На что смотреть:

- `speaker_id` есть у всех сегментов или стоят `quality_flags`;
- нет ли потерь на границах чанков (около 15, 30, 45 минут);
- корректны ли технические термины и английские вставки;
- `corrections` содержат только осмысленные замены.

### 7. Имена говорящих (вручную)

Имена никогда не угадываются автоматически.

```bash
meeting-transcriber speakers data/output/meeting.json
```

Команда печатает время речи, число реплик и 3 примера на говорящего, а также
создаёт шаблон:

```text
data/output/meeting.speakers.yaml
```

Заполните `name`:

```yaml
speakers:
  speaker_00:
    name: Иван
  speaker_04:
    name: Мария
```

Примените имена без повторного ASR:

```bash
meeting-transcriber label-speakers data/output/meeting.json \
  --map data/output/meeting.speakers.yaml
```

В Markdown будет `Иван (speaker_00)`, в SRT — `[Иван]`, в JSON —
`display_name`; `speaker_id` сохраняется для аудита.

### 8. Протокол совещания (llama.cpp)

```bash
./scripts/start_llama_server.sh

meeting-transcriber report data/output/meeting.json \
  --map data/output/meeting.speakers.yaml \
  --config config/default.yaml

./scripts/stop_llama_server.sh
```

Если имена ещё не заполнены, просто уберите `--map`.

Результаты:

```text
data/output/meeting.report.json   # структурированный протокол + ссылки S<n>
data/output/meeting.report.md     # отдельный Markdown-протокол
data/output/meeting.markdown      # расшифровка + протокол в конце
```

Ориентиры: первый проход по 87-минутной записи — **~8–10 минут** (20 чанков +
консолидация тем + резюме). Повторный запуск использует кэш чанков и занимает
**~2–3 минуты**. Модель всегда выгружается командой
`./scripts/stop_llama_server.sh`.

### 9. Повторные запуски, кэш и восстановление

- `work/<meeting>/` хранит артефакты стадий и `state.json`.
- Повторный `transcribe` без `--force` переиспользует готовые стадии.
- Изменение настроек инвалидирует только затронутые стадии (например, правка
  глоссария не перезапускает ASR).
- `--force` пересчитывает всё и перезаписывает выходные файлы.
- Без `--force` существующие выходные файлы не перезаписываются: создаётся
  копия с суффиксом времени.

Если `data/output/` случайно очищен, но `work/` цел, восстановление занимает
секунды:

```bash
meeting-transcriber transcribe data/input/meeting.webm \
  --config config/default.yaml --skip-llm

./scripts/start_llama_server.sh
meeting-transcriber report data/output/meeting.json --config config/default.yaml
./scripts/stop_llama_server.sh
```

Полный пересчёт нужен только если удалён `work/<meeting>/` или изменён
исходный файл.

### 10. Видео и несколько аудиодорожек

```bash
# посмотреть дорожки
ffprobe -hide_banner -show_streams data/input/meeting.mp4

# извлечь дорожку отдельно (опционально)
meeting-transcriber extract-audio data/input/meeting.mp4 \
  --output data/output/meeting.mka --audio-stream 1 --force

# транскрибировать нужную дорожку
meeting-transcriber transcribe data/input/meeting.mp4 \
  --config config/default.yaml --skip-llm --audio-stream 1
```

FFmpeg копирует дорожку без перекодирования в `audio_original.mka`
(при необходимости — FLAC fallback), затем делает 16 кГц mono WAV.

### 11. Docker: сборка и использование

#### 11.1. Что нужно

- Docker Engine 24+ и Compose v2;
- NVIDIA driver + `nvidia-container-toolkit`;
- модели уже скачаны на хосте (`models/`, ~7.2 ГиБ);
- свободное место: образ **8.14 ГБ** + build cache + модели; удобно иметь 25–30 ГБ;
- интернет **только на этапе сборки** (Docker Hub, PyPI). Runtime работает offline.

Проверка GPU для Docker:

```bash
docker info | grep -i nvidia
docker run --rm --gpus all pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime \
  python -c "import torch; print(torch.cuda.is_available())"
```

#### 11.2. Сборка образа

```bash
cd /home/demakov/meeting-transcriber
./scripts/build_image.sh
```

Скрипт:

1. создаёт `work/`, `data/output/`, `logs/`;
2. собирает `docker/Dockerfile`;
3. ставит тег `meeting-transcriber:0.1.0`.

Свой тег:

```bash
IMAGE=meeting-transcriber:2026-11 ./scripts/build_image.sh
```

Полная пересборка без кэша:

```bash
./scripts/build_image.sh --no-cache
```

Что происходит внутри: базовый `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`
→ FFmpeg → `requirements/asr.txt` + `requirements/diarization.txt` → установка
пакета → non-root пользователь. **Модели в образ не копируются.** Build context
маленький: `.dockerignore` исключает `.venv*`, `models/`, `data/`, `work/`,
`tests/`, `docs/`.

Ориентиры: размер **8.14 ГБ**, время сборки 15–40 минут в зависимости от сети.

#### 11.3. Проверка образа

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

`scripts/compose.sh` — обязательная обёртка: она экспортирует `HOST_UID` и
`HOST_GID`, создаёт каталоги и запускает `docker compose` с правильным
пользователем. Прямой `docker compose run` использует uid 1000 из образа; на
хосте с другим uid это даст `permission denied` при записи в `work/` и
`data/output/`.

#### 11.4. Запуск транскрипции

```bash
cp "/путь/к/записи.webm" data/input/meeting.webm

./scripts/compose.sh run --rm transcriber transcribe \
  /data/input/meeting.webm \
  --config /app/config/docker.yaml \
  --skip-llm
```

Полезные опции:

```bash
# точное число участников
--num-speakers 12

# верхняя граница автоопределения
--max-speakers 20

# дорожка в многодорожечном видео
--audio-stream 1

# перезаписать результаты и пересчитать стадии
--force
```

`--skip-llm` recommended: LLM-протокол в контейнере отключён
(`network_mode: "none"`), а ASR не должен конкурировать с `llama-server` за
VRAM. Первый контейнерный прогон 87-минутной записи: **~2m40s**, 8 говорящих,
пик **4612 MiB**.

Результаты появятся на хосте:

```text
data/output/meeting.json
data/output/meeting.markdown
data/output/meeting.srt
```

Если файлы уже существуют, без `--force` создаются копии с суффиксом времени.
Во `work/<meeting>/` сохраняются стадии и кэш; повторный запуск продолжит с
последней завершённой стадии.

#### 11.5. Что монтируется

| Хост | Контейнер | Режим | Назначение |
|---|---|---|---|
| `./models` | `/models` | `ro` | веса + `manifest.yaml` / lock |
| `./data/input` | `/data/input` | `ro` | исходные аудио/видео |
| `./data/output` | `/data/output` | `rw` | JSON/Markdown/SRT/протокол |
| `./work` | `/work` | `rw` | чанки, стадии, resume |
| `./config` | `/app/config` | `ro` | конфиги и глоссарии |

Образ не содержит весов и не скачивает их. `verify-models` берёт манифест из
`/models/manifest.yaml`.

#### 11.6. Протокол совещания

В контейнере LLM выключен намеренно: `compose.yaml` использует
`network_mode: "none"`, а модель занимает ~6.7 ГБ VRAM. Протокол генерируется
нативно на хосте по уже смонтированным результатам:

```bash
./scripts/start_llama_server.sh
meeting-transcriber report data/output/meeting.json \
  --map data/output/meeting.speakers.yaml \
  --config config/default.yaml
./scripts/stop_llama_server.sh
```

Если хотите LLM в контейнере, переключите Compose на bridge-сеть и поднимите
`llama-server` на хосте, затем выставьте `llm.enabled: true` и
`llm.base_url: http://host.docker.internal:8080/v1` в `config/docker.yaml`.
На GPU 8 ГБ это требует ручной синхронизации: ASR и llama-server не должны
работать одновременно.

#### 11.7. Перенос на офлайн-машину

На машине с интернетом:

```bash
docker save meeting-transcriber:0.1.0 | gzip > meeting-transcriber-image.tar.gz
tar czf meeting-transcriber-models.tar.gz models/
```

На целевой машине:

```bash
gunzip -c meeting-transcriber-image.tar.gz | docker load
tar xzf meeting-transcriber-models.tar.gz

cd /home/demakov/meeting-transcriber
./scripts/compose.sh run --rm transcriber transcribe \
  /data/input/meeting.webm \
  --config /app/config/docker.yaml \
  --skip-llm
```

Если репозитория на целевой машине нет, можно запускать образом напрямую:

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

#### 11.8. Обновление и очистка

После изменения кода/зависимостей:

```bash
./scripts/build_image.sh
```

Модели и результаты при этом не затрагиваются — они на хосте. Удалить старый
образ и освободить build cache:

```bash
docker image rm meeting-transcriber:0.1.0
docker builder prune
docker system df
```

#### 11.9. Docker troubleshooting

| Симптом | Причина / решение |
|---|---|
| `permission denied` в `work/` или `data/output/` | запускать через `./scripts/compose.sh`, а не `docker compose` напрямую |
| `could not select device driver ... gpu` | не установлен `nvidia-container-toolkit` |
| `CUDA out of memory` | на GPU висит `llama-server`; `./scripts/stop_llama_server.sh` |
| `cannot cache function '__o_fold'` | в контейнере нет writable-кэша numba; используйте `compose.sh` (он задаёт env) |
| `model not found` / пустой `/models` | не смонтирован `models/` или нет `models/manifest.yaml` |
| `network_mode: "none"` и `report` | это ожидаемо; запускайте протокол нативно |
| сборка падает на `pip install` | проверьте доступ к PyPI и свободное место |

### 12. Что влияет на качество

| Что | Где | Эффект |
|---|---|---|
| `--num-speakers N` | CLI | точное число говорящих вместо автоопределения |
| `--max-speakers N` | CLI / YAML | верхняя граница, по умолчанию 20 |
| `terminology.profiles` | YAML | словарь терминов под проект |
| `terminology.threshold` | YAML | выше — меньше замен, но и меньше правок |
| `llm.max_topics` | YAML | сколько крупных тем оставить (по умолчанию 12) |
| `llm.chunk_chars` | YAML | размер чанка для LLM; меньше — стабильнее по контексту |
| `LLAMA_CTX` | env при старте | контекст llama-server (по умолчанию 8192) |

Если отчёт слишком дробный — уменьшите `max_topics`; если в отчёте не хватает
деталей — увеличьте. Если термины не исправляются — добавьте их в
`config/terminology/*.yaml` и перезапустите только стадию терминологии
(без `--force` ASR переиспользуется).

### 13. Быстрый troubleshooting

| Симптом | Что делать |
|---|---|
| `command not found: meeting-transcriber` | `source .venv/bin/activate` |
| `ffmpeg not found` | `./scripts/install_host_dependencies.sh` |
| `ASR model directory not found` | `./scripts/download_models.sh --only asr` |
| `speaker embedding checkpoint not found` | `./scripts/download_models.sh --only diarization` |
| CUDA OOM | остановить `llama-server`; уменьшить `batch_size`/`beam_size` |
| `llama-server` не отвечает | `./scripts/start_llama_server.sh`, затем `tail logs/llama-server.log` |
| пустой протокол / `reasoning_content` | сервер должен быть запущен с `--reasoning off` (это делает скрипт) |
| отчёт оборвался на консолидации | уменьшить `llm.merge_chars` или поднять `LLAMA_CTX` |
| « Speaker_00 » в тексте отчёта | обновить код/перегенерировать `report` |

Полный список: [`docs/troubleshooting.md`](docs/troubleshooting.md).

### 14. Ограничения

- Diarization не разделяет одновременную речь, только помечает перекрытие.
- `speaker_XX` — не имя человека; имена задаются вручную.
- WER/DER на эталонной записи не измерялись.
- Docker-образ в этом окружении не собирался.
- LLM-протокол — черновик: проверяйте ссылки `S<n>` и спорные пункты.

## Video input (FFmpeg)

Any container FFmpeg understands is accepted: MP4, MKV, MOV, WebM, AVI, TS, and
plain audio files. The pipeline:

1. probes streams with `ffprobe` and validates the requested audio track;
2. copies the selected track losslessly to
   `work/<meeting>/audio_original.mka` (`-vn -map 0:a:N -c:a copy`, FLAC
   fallback if the codec cannot be muxed);
3. converts that track to `audio_16k_mono.wav` for the models.

Standalone extraction (useful for checking or archiving the track):

```bash
# default track 0 -> data/output/meeting.mka
meeting-transcriber extract-audio data/input/meeting.mp4

# explicit output and audio stream
meeting-transcriber extract-audio data/input/meeting.mp4 \
  --output data/output/meeting.mka --audio-stream 1 --force

# list tracks
ffprobe -hide_banner -show_streams data/input/meeting.mp4
```

For a multi-track recording pass `--audio-stream N` to `transcribe` (or set
`audio.audio_stream` in YAML). Set `audio.keep_original: false` to skip the
intermediate `.mka`; the audio is still extracted internally by FFmpeg.
Metadata (`has_video`, `video_codec`, `audio_streams`, `audio_stream_index`,
`extracted_audio`, SHA-256) is stored in `preprocess.json` and the final JSON.

## Docker

The detailed Russian walkthrough is in
[«Пошаговый гайд → 11. Docker»](#11-docker-сборка-и-использование). Short
reference:

### Build

```bash
# one-time: models on the host
./scripts/download_models.sh

# build the image (~8.1 GB, 15–40 min, network required)
./scripts/build_image.sh

# custom tag / clean rebuild
IMAGE=meeting-transcriber:2026-11 ./scripts/build_image.sh
./scripts/build_image.sh --no-cache
```

The image does **not** contain model weights. `.dockerignore` excludes
`.venv*`, `models/`, `data/`, `work/`, `tests/`, and `docs/`, so the build
context is small.

### Verify

```bash
./scripts/compose.sh run --rm transcriber --help
./scripts/compose.sh run --rm transcriber validate-config \
  --config /app/config/docker.yaml
./scripts/compose.sh run --rm transcriber verify-models \
  --config /app/config/docker.yaml
./scripts/compose.sh run --rm transcriber python -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Run

```bash
./scripts/compose.sh run --rm transcriber transcribe \
  /data/input/meeting.webm \
  --config /app/config/docker.yaml \
  --skip-llm
```

Use `./scripts/compose.sh` rather than plain `docker compose`: it exports
`HOST_UID`/`HOST_GID` so bind-mounted `work/` and `data/output/` are writable,
and creates those directories. A plain `docker compose` run defaults to
`uid 1000` and can fail on files owned by another host user.

Compose mounts `models/` read-only at `/models`, `data/input` read-only,
`data/output` and `work/` writable, and uses `network_mode: "none"`. The LLM
report is disabled in the container (no network); generate it natively on the
host from the mounted `data/output` (see
[Meeting minutes](#meeting-minutes-local-llm-llamacpp)).

### Offline transfer

```bash
# connected host
docker save meeting-transcriber:0.1.0 | gzip > meeting-transcriber-image.tar.gz
tar czf meeting-transcriber-models.tar.gz models/

# offline target
gunzip -c meeting-transcriber-image.tar.gz | docker load
tar xzf meeting-transcriber-models.tar.gz
./scripts/compose.sh run --rm transcriber transcribe \
  /data/input/meeting.webm --config /app/config/docker.yaml --skip-llm
```

## CLI

```text
meeting-transcriber transcribe <audio-or-video-file> [options]
meeting-transcriber extract-audio <video-file> [--output FILE.mka] [--audio-stream N]
meeting-transcriber speakers <transcript.json> [--template FILE] [--force]
meeting-transcriber label-speakers <transcript.json> --map <speakers.yaml>
meeting-transcriber report <transcript.json> [--map <speakers.yaml>]
meeting-transcriber verify-models
meeting-transcriber list-models
meeting-transcriber download-models [--only asr|diarization|llm|all]
meeting-transcriber validate-config
meeting-transcriber build-glossary <project-path>
```

Transcription options:

| Option | Meaning |
| --- | --- |
| `--config PATH` | YAML configuration (default: `config/default.yaml`) |
| `--language ru` | explicit ASR language |
| `--output-dir DIR` | override `paths.output_dir` |
| `--models-dir DIR` | override `paths.models_dir` |
| `--profile NAME` | use one terminology profile |
| `--num-speakers N` | known participant count (oracle mode) |
| `--max-speakers N` | upper bound for automatic counting (default 20) |
| `--skip-diarization` | ASR only |
| `--skip-llm` | no meeting report |
| `--speaker-map FILE` | apply display names before report/output |
| `--device cuda\|cpu\|auto` | override device |
| `--audio-stream N` | 0-based audio track index for multi-track videos |
| `--force` | ignore cached stages, overwrite outputs |

`build-glossary` scans an explicitly provided C++ repository (classes, structs,
namespaces, types, functions, CMake targets) and writes a YAML glossary. It never
scans a path you did not pass.

## Speaker names

Speaker ids stay anonymous and are never guessed. Review them and assign names
manually:

```bash
# Prints talk time, segment count and 3 sample quotes per speaker.
# Also writes data/output/meeting.speakers.yaml with empty name fields.
meeting-transcriber speakers data/output/meeting.json

# Fill the "name" fields, then rewrite JSON/Markdown/SRT without re-running ASR:
meeting-transcriber label-speakers data/output/meeting.json \
  --map data/output/meeting.speakers.yaml
```

The Markdown shows `Иван (speaker_00)`, the SRT shows `[Иван]`; the JSON keeps
`speaker_id` for every segment and adds `display_name` to the speaker list. A map
can also be passed directly to `transcribe --speaker-map`.

## Meeting minutes (local LLM, llama.cpp)

The protocol is generated by **llama.cpp** (`llama-server`) with the unsloth
**Qwen3.5-9B UD-Q4_K_XL** GGUF. Everything runs locally.

```bash
# 1. Install a pinned llama.cpp CUDA build into ~/.local (no sudo)
./scripts/install_llama_cpp.sh

# 2. Download the model (~5.6 GiB)
./scripts/download_models.sh --only llm

# 3. Start the OpenAI-compatible server on 127.0.0.1:8080
./scripts/start_llama_server.sh

# 4. Build the protocol from an existing transcript (optional names)
meeting-transcriber report data/output/meeting.json \
  --map data/output/meeting.speakers.yaml

# 5. Free VRAM before the next ASR run
./scripts/stop_llama_server.sh
```

Outputs:

```text
data/output/meeting.report.json   # structured report with S<n> references
data/output/meeting.report.md     # standalone protocol
data/output/meeting.markdown      # transcript + appended protocol
```

Long transcripts are processed hierarchically (map-reduce): chunks of
`llm.chunk_chars` are summarised, aggregated deterministically (near-duplicates
merged; distinct decisions/tasks preserved), then topics are consolidated into
`llm.max_topics` themes. The LLM is asked only for the final title and summary.
Every `S<n>` reference is validated; unknown ones are dropped. `--skip-llm`
disables the report for one run.

The model is served with `--reasoning off` (Qwen3.5 thinking would consume the
JSON output budget) and an 8k context. On an 8 GB GPU the server occupies about
6.7 GiB, so stop it before running ASR again.

## Configuration

`config/default.yaml` is fully commented and validated. Unknown keys are
rejected. All relative paths resolve against `paths.root` (repository root by
default). Environment substitution is available as `{env:NAME}`.

Key settings:

```yaml
audio:
  keep_original: true   # copy the video's audio track to audio_original.mka
  audio_stream: 0       # 0-based track index for multi-audio videos

asr:
  model_dir: whisper-large-v3-turbo-ct2
  device: cuda
  compute_type: float16
  beam_size: 5
  chunk_seconds: 900
  overlap_seconds: 15

diarization:
  speaker_model: speakerverification_en_titanet_large.nemo
  max_speakers: 20
  num_speakers: null       # set to force an exact count

terminology:
  profiles: [general-it]
  threshold: 0.86
  apply_corrections: true  # false = suggestions only

llm:
  enabled: true
  base_url: http://127.0.0.1:8080/v1
  model: qwen3.5-9b
  chunk_chars: 5000       # map-reduce chunk size
  merge_chars: 12000      # consolidation input limit
  max_topics: 12          # consolidate granular topics
  num_ctx: 8192           # llama-server context
```

## Models

| Stage | Model | Revision | License |
| --- | --- | --- | --- |
| ASR | `deepdml/faster-whisper-large-v3-turbo-ct2` | `4df90f7` | MIT |
| VAD | Silero VAD 6.2.3 (wheel) | — | MIT |
| Embeddings | `nvidia/speakerverification_en_titanet_large` | `0dc382f` | CC-BY-4.0 |
| Report LLM | `unsloth/Qwen3.5-9B-GGUF` (UD-Q4_K_XL) | `3885219` | Apache-2.0 |

Details, sizes, and alternatives: [`models/README.md`](models/README.md) and
[`docs/model-selection.md`](docs/model-selection.md).

## Repository layout

```text
meeting-transcriber/
├── config/                 # default/docker configs + terminology profiles
├── data/input, data/output # volumes (gitignored contents)
├── docker/                 # Dockerfile + entrypoint
├── docs/                   # architecture, model selection, offline, GPU, quality
├── models/                 # manifest + README (weights gitignored)
├── requirements/           # base / asr / diarization / dev
├── scripts/                # install, download, verify, build, offline smoke
├── src/meeting_transcriber/
│   ├── audio/              # ffprobe metadata, mono 16k conversion, chunking
│   ├── asr/                # Transcriber interface + faster-whisper backend
│   ├── diarization/        # Diarizer interface + NeMo + Silero VAD
│   ├── alignment/          # temporal fusion
│   ├── terminology/        # glossary, matcher, correction, C++ importer
│   ├── postprocessing/     # local LLM client + meeting report
│   ├── pipeline/           # runner, state, persistence
│   ├── output/             # JSON, Markdown, SRT writers
│   ├── cli.py, config.py, models.py, types.py
├── tests/                  # unit + integration (no GPU needed)
├── compose.yaml
└── pyproject.toml
```

## Testing

Unit and integration tests use fake backends and need no GPU, FFmpeg, or weights:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
ruff check src tests
```

Real-model smoke test:

```bash
./scripts/offline_smoke_test.sh data/input/smoke.wav
```

## Offline guarantees

* model directories and `.nemo` files are required to exist before inference;
* missing weights exit with code 3 and a hint, never a download;
* `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set in the backend, image,
  Compose, and smoke test;
* Compose runs with `network_mode: "none"`;
* `scripts/offline_smoke_test.sh` uses a private network namespace when available.

See [`docs/offline-deployment.md`](docs/offline-deployment.md).

## Documentation

* [Architecture and ADRs](docs/architecture.md)
* [Model selection and research evidence](docs/model-selection.md)
* [Offline deployment](docs/offline-deployment.md)
* [GPU memory and sequential execution](docs/gpu-memory.md)
* [Quality evaluation](docs/quality-evaluation.md)
* [Troubleshooting](docs/troubleshooting.md)
* [Known limitations](docs/known-limitations.md)

## Status

The MVP is implemented and covered by 61 unit/integration tests. End-to-end run
on an 87-minute WebM (Russian, mixed English terms): 945 segments, 8 speakers,
peak VRAM 4612 MiB, wall time 2m31s on an RTX 4060 8 GB. The final protocol uses
unsloth Qwen3.5-9B UD-Q4_K_XL via llama.cpp and consolidated the meeting into
12 topics, 6 decisions and 10 action items. The Docker image builds (8.14 GB)
and passes containerized ASR + diarization on the full 87-minute recording
(918 segments, 8 speakers, 2m40s, peak 4612 MiB); the optional LLM report is
intentionally run on the host because Compose uses `network_mode: "none"`.
WER/DER against ground truth were not measured.

## License

Code: MIT. Model weights retain their upstream licenses (MIT, CC-BY-4.0,
Apache-2.0).
