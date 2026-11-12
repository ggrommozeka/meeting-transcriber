# План сборки: Strata + ISTA-DASLab Coder IQ1_M на Debian

Цель: локальный кодинг-сервер с контекстом **262 144 токена** на хосте
2 × RTX 5060 Ti 16 GB + 80 GB RAM, совместимый по API с текущим пайплайном
(`llm.base_url: http://127.0.0.1:8080/v1`).

Почему именно этот пак:

| Критерий | Значение |
| --- | --- |
| Модель | Qwen3.8-Flash-Next, ISTA-DASLab GSQ-RCO Coder |
| Эксперты | 256 из 512 на слой (отобраны на коде), 29.6 GB |
| Качество | ~91% SWE-bench Verified и ~99% LiveCodeBench от полной модели |
| Слабые места | не-код, не-английский/китайский текст |
| KV на 262K | ~2.7–3.5 GB (всего 12 full-attention слоёв) |
| Ожидаемый decode | 60–90 tok/s, prefill ~1.5–2К tok/s (~2.5 мин на 260K) |

Движок: **Strata 0.2.13** (Terminal + engine 0.1.59), режим без root.
Причина: это единственный движок, который по умолчанию держит 29 GB
n-gram/PLE-таблицу на SSD и кэширует экспертов в VRAM обоих GPU.

> Если понадобится универсальная модель (русский язык, саммари встреч),
> повторяется тот же план с `--family qwen --model IQ3_XXS` (47 GB, 262K
> тоже влезает). Coder IQ1_M — только для кодинга.

---

## 0. Предпосылки и проверка железа

- [ ] Debian 12/13, ядро 6.1+, пользователь в группе `video` (для `nvidia-smi`).
- [ ] NVIDIA-драйвер **≥ 580** (engine собран под CUDA 13). Проверить:
      `nvidia-smi` → `Driver Version`.
- [ ] Обе карты видны и не заняты:
      `nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv`
- [ ] 80 GB RAM доступно (`free -g`, колонка `available`).
- [ ] **≥ 90 GB свободно на NVMe** под модели (`df -h /srv`). 29 GB engram
      читается с диска во время работы — SATA/HDD дадут провалы.
- [ ] Пакеты: `sudo apt install -y curl ca-certificates openssl tar python3-venv libgomp1`
      (`setup.sh` тоже умеет предлагать `apt`, но лучше поставить заранее).

Опционально, чтобы движок не терял скорость из-за драйвера:

```bash
sudo nvidia-smi -pm 1                    # persistence mode
sudo systemctl enable --now nvidia-persistenced
```

Если engine в логе ругается на невозможность закрепить память:

```bash
ulimit -l unlimited                      # memlock для pinned RAM
```

---

## 1. Установка Strata Terminal (без root)

```bash
cd /tmp
curl -fsSL --connect-timeout 15 --max-time 120 \
  https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh
sh install.sh
```

- Ставит `strata` в PATH (`~/.local/bin`), права администратора не нужны.
- Engine (часть, работающая с GPU) **не входит** в инсталлятор: он скачивается
  при первом `strata launch`, проверяется по подписи и тест-запускается.

Проверка:

```bash
strata --version
strata launch --help          # зафиксировать фактические имена флагов
```

> **Важно:** флаги движка (`--set KEY=VALUE`) в 0.2.13 именуются в kebab-case.
> Перед шагом 3 сверить точные имена с `strata launch --help` и с
> `strata launch --plan ...` — план печатает итоговую командную строку engine.

### Вариант B: Docker (если нужна изоляция)

```bash
docker run -d --name strata --gpus all -p 127.0.0.1:8080:8080 \
  --ulimit memlock=-1:-1 --shm-size 2g --stop-timeout 120 \
  -v strata-data:/data \
  -e FAMILY=coder -e MODEL=IQ1_M -e API_KEY=$(openssl rand -hex 24) \
  ghcr.io/niko1221/strata:0.2.7-cuda13
```

Ограничения: контекст/KV задаются не флагами `strata launch`, а через
конфиг в volume; `--context` в CLI недоступен. Для 262K предпочтителен
нативный вариант (шаги 1–3), Docker — резервный.

---

## 2. План и загрузка модели (dry-run сначала)

```bash
strata launch --plan \
  --family coder --model IQ1_M \
  --gpu 0,1 \
  --context 262144 \
  --kv k8v4 \
  --vision off \
  --data-dir /srv/strata
```

Что проверить в выводе плана:

- [ ] `fits` (не `tight`, не `does not fit`) для 32 GB VRAM + 80 GB RAM.
- [ ] Объём загрузки: шарды модели + MTP draft (~5.2 GB), vision не качается
      (`--vision off`). Ориентир: **~65–75 GB**, точную цифру даёт план.
- [ ] Каталог данных на NVMe.
- [ ] Роли карт: при двух одинаковых картах Main = карта с большим индексом,
      Helper = вторая. При необходимости зафиксировать:
      `--gpu-roles 1=main,0=helper`.

Только после этого — реальный запуск (скачивание идёт десятки минут):

```bash
strata launch \
  --family coder --model IQ1_M \
  --gpu 0,1 \
  --context 262144 \
  --kv k8v4 \
  --vision off \
  --parallel 1 \
  --data-dir /srv/strata \
  --access lan \
  --save-preset "Coder IQ1_M 262k" \
  --yes --no-tui
```

- `--access lan` — только если контейнер `meeting-transcriber` (`network_mode:
  none`) или другой хост должен достучаться до API. Иначе убрать и оставить
  `127.0.0.1`.
- `--save-preset` фиксирует все параметры; дальше запуск одной строкой:
  `strata launch --preset "Coder IQ1_M 262k"`.

---

## 3. Тюнинг под 2×16 GB и полный контекст

Базовые значения уже заданы флагами выше. Дополнительно — параметры engine
через `--set` (сверить имена в `--help`):

| Параметр | Значение | Зачем |
| --- | --- | --- |
| `kv-streaming=on` | принудительно | KV живёт в RAM, в VRAM только читаемое окно. С 64K включается автоматически, но `on` исключает «не хватило RAM» |
| `kv-resident=32768` | 32K токенов | сколько KV держать в VRAM. При 262K это ~0.5 GB VRAM вместо ~3 GB |
| `vram-reserve-mib=1024` | 1 GB | иначе нулевой свободный VRAM → драйвер выгружает экспертов, скорость падает в разы |
| `prompt-cache-every=16384` | 16K | чекпоинты промпта для агентских циклов (контекст переотправляется каждый шаг) |
| `expert-cache=auto` | по умолчанию | ~18–20 GB VRAM под кэш экспертов при 32K resident KV |
| `mtp=on` | по умолчанию | draft-слой даёт 1.3–1.7× decode |

Чего **не** делать:

- **Не включать KVA** (`STRATA_KVA`/«Faster long prompts»). Это аппроксимация
  поздних слоёв; на кодинг-агентах замеры дают 0/10 и 6/10 прогонов против
  4–7/10 без аппроксимации. Теряются строки и арифметика.
- **Не ставить `--parallel` > 1.** Каждый слот умножает KV: +3.5 GB RAM и
  −700 экспертов в VRAM на слот.

Проверить, что применилось:

```bash
strata launch --status
strata launch --list-presets
```

В мониторе/статусе должны быть: `context 262144`, `kv k8v4`, `streaming on`,
две карты, непустой `expert_slots`.

---

## 4. Приёмка

### 4.1 API живой и совместимый

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/models
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"strata","messages":[{"role":"user","content":"def f(x):"}],"max_tokens":32}'
```

### 4.2 Контекст действительно 262K (needle-тест)

Сгенерировать ~200K токенов «шума» с контрольной строкой в середине и
попросить её вернуть. Критерии приёмки:

- [ ] запрос не отклонён по длине контекста;
- [ ] TTFT ≈ 200K / prefill ≈ **1.5–2 мин**;
- [ ] needle найден;
- [ ] в логе engine нет `WARNING: only N MiB of VRAM free`.

### 4.3 Скорость

Зафиксировать и записать в `docs/gpu-memory.md` (по образцу существующей
таблицы замеров):

| Метрика | Ожидание | Факт |
| --- | --- | --- |
| prefill 4K | 1.2–1.6К tok/s | |
| prefill 64K | 1.6–1.9К tok/s | |
| prefill 200K | 1.5–1.8К tok/s | |
| decode 4K | 70–90 tok/s | |
| decode 200K | 45–65 tok/s | |
| RSS процесса | ≤ 45 GB | |
| VRAM обеих карт | ≤ 31 GB | |

### 4.4 Откат

- `strata launch --stop` — штатная остановка.
- Модель/данные: `/srv/strata` (удаляется по желанию).
- Прежний путь отчётов не трогается: `scripts/start_llama_server.sh`
  (Qwen3.5-9B, 8K) остаётся рабочим, пока порт 8080 занят не им — см. ниже.

---

## 5. Интеграция с meeting-transcriber

Код менять не нужно: `src/meeting_transcriber/postprocessing/llm_client.py`
ходит в любой OpenAI-совместимый `/v1/chat/completions` через `requests`.

- [ ] В `config/default.yaml` для кодинг-сценария:
      `llm.base_url: http://127.0.0.1:8080/v1`, `llm.model: strata`.
      Для отчётов по встречам оставить отдельный профиль с Qwen3.5-9B —
      Coder-пак слаб на русском и не-коде.
- [ ] Порог 8080: одновременно Strata и `llama-server` не поднять. Перед
      ASR-этапом останавливать Strata (`strata launch --stop`), как сейчас
      делается для llama.cpp в `scripts/stop_llama_server.sh`.
- [ ] Добавить (отдельной задачей, не в этом плане) `scripts/start_strata.sh`
      / `scripts/stop_strata.sh` по образцу существующих, с health-check и
      `--context`/`--kv` из переменных окружения.
- [ ] Для Docker-пути: `config/docker.yaml` → `base_url:
      http://host.docker.internal:8080/v1` + `enabled: true` и bridge-сеть
      вместо `network_mode: none`.

---

## 6. Риски и что делать

| Симптом | Причина | Действие |
| --- | --- | --- |
| `does not fit` в плане | карты не обе видны или заняты | `nvidia-smi`, закрыть другие CUDA-процессы |
| decode упал в 5–7 раз после прогрева | 0 MiB свободной VRAM | `vram-reserve-mib=1536`, уменьшить `expert-cache` |
| OOM при загрузке | engram/PLE на медленном диске или мало RAM | освободить RAM, проверить `df -h`, NVMe |
| KV-стриминг не включился | «нет места в RAM» по оценке setup | `--set kv-streaming=on` (override) |
| Медленный prefill на 262K | KVA выключен — это нормально | не включать KVA ради кодинга; держать важное в хвосте промпта |
| Модель путается в середине 250K | разреженное QSA-внимание | держать стек/диффы/инструкции в конце, включить `prompt-cache-every` |

---

## 7. Порядок выполнения (чек-лист)

1. [ ] Драйвер ≥ 580, обе карты, 80 GB RAM, ≥ 90 GB NVMe — проверено.
2. [ ] `install.sh` → `strata --version` OK.
3. [ ] `strata launch --plan ...` → `fits`, объём загрузки зафиксирован.
4. [ ] Реальная загрузка + `--save-preset`.
5. [ ] `--set` для `kv-streaming`, `kv-resident`, `vram-reserve-mib`,
       `prompt-cache-every`; KVA выключен; `--parallel 1`.
6. [ ] `/health`, `/v1/models`, короткий запрос — OK.
7. [ ] Needle на 200K — найден, TTFT в норме, лог без VRAM-warning.
8. [ ] Замеры внесены в `docs/gpu-memory.md`.
9. [ ] Профиль `llm` для кодинга отделён от профиля отчётов встреч.
10. [ ] (опционально) скрипты `start_strata.sh` / `stop_strata.sh`.
