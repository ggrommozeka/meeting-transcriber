# Architecture

## Overview

```text
audio file
   │
   ▼
[audio.preprocess]  ffprobe streams → lossless track extraction (video) →
                     FFmpeg 16 kHz mono WAV, 15-min chunks (15 s overlap)
   │
   ▼
[asr.whisper_backend]  faster-whisper / CTranslate2, local model directory
   │  raw segments + word timestamps
   ▼
[diarization.nemo_backend]  Silero VAD → NeMo TitaNet-Large → spectral clustering
   │  speaker intervals (RTTM parsed to JSON)
   ▼
[alignment.fusion]  interval intersection, word-level assignment, split at speaker changes
   │
   ▼
[terminology]  glossary → exact/fuzzy matching → auditable corrections
   │
   ▼
[speakers]  optional manual speaker_id → display_name map
   │
   ▼
[postprocessing.llm]  optional local OpenAI-compatible endpoint
   │  map-reduce: chunk summaries → merged meeting minutes
   ▼
outputs: JSON (canonical) + Markdown + SRT + report.json/report.md
```

Heavy stages run **sequentially** on one GPU. The ASR model is released before
diarization starts, and vice versa. Nothing keeps both models resident.

## Components and contracts

| Component | Interface | Implementation |
| --- | --- | --- |
| ASR | `asr.base.Transcriber` | `asr.whisper_backend.WhisperBackend` |
| Diarization | `diarization.base.Diarizer` | `diarization.nemo_backend.NemoDiarizer` |
| Terminology | `terminology.correction.TerminologyCorrector` | glossary + matcher |
| LLM | `postprocessing.llm_client.LLMClient` | OpenAI-compatible HTTP |
| Output | writer functions | JSON / Markdown / SRT |

All stage outputs are plain dataclasses in `types.py`. `Transcript.to_dict()`
is the versioned output schema (`schema_version: "1.0"`).

## Data flow and resume

Each meeting gets an isolated work directory:

```text
work/<meeting_id>/
├── audio_original.mka        # lossless audio track extracted from video
├── audio_16k_mono.wav
├── audio_source.json         # provenance: source SHA-256 + selected stream
├── chunks/chunk_0000.wav ...
├── preprocess.json
├── asr.json
├── diarization.json
├── transcript_aligned.json
├── transcript_corrected.json
├── report.json
├── transcript.json
├── diarizer/                 # NeMo intermediate files
└── state.json
```

`state.json` stores a fingerprint of the effective configuration. A stage is
reused only when the fingerprint matches, the stage is marked completed, and its
artifact exists. Changing any configuration option invalidates cached stages;
audio files themselves are reused. `--force` ignores all cached stages.

The original audio is never written to. ASR output (`raw_text`) is never
mutated; corrections live in `corrected_text` and `corrections`.

## Offline guarantees

* `WhisperBackend` requires an existing local model directory and rejects
  incomplete directories before importing CTranslate2.
* `NemoDiarizer` requires a local `.nemo` file and never uses a model name.
* Silero VAD ships inside its wheel.
* `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set defensively at model
  load time and in Docker/Compose.
* The downloader is the only component that performs network I/O, and only when
  explicitly invoked.
* `docker compose` uses `network_mode: "none"`.

## Deliberate non-goals

No web UI, database, message broker, or always-on service. This is a batch CLI.

## Architecture decisions (ADR)

### ADR-001: faster-whisper over whisper.cpp

Chosen because:

* native CTranslate2 models are small and fast on 8 GB GPUs;
* word-level timestamps, VAD filtering, and `initial_prompt` are supported
  first-class;
* a local directory can be passed directly to `WhisperModel`, which gives a
  hard offline boundary.

whisper.cpp remains a viable alternative behind the `Transcriber` interface.

### ADR-002: NeMo cascade instead of a 4-speaker end-to-end model

Sortformer-style end-to-end diarizers are limited to a fixed speaker count and
do not match the 8–15 participant requirement. The NeMo cascade
(VAD → TitaNet embeddings → spectral clustering) supports
`max_num_speakers` up to 20 in this configuration and auto-counts speakers.
It is not overlap-aware; overlapping speech is reported through
`overlap_speakers` but not separated.

Because NeMo's clustering builds an O(N²) similarity matrix, recordings longer
than `diarization.long_audio_threshold_seconds` (default 30 min) automatically
switch to a single-scale embedding grid (`1.5 s / 0.75 s`). On an 87-minute
recording this reduced the run from an out-of-memory failure at ~7 GiB to a
35-second diarization stage with peak process VRAM below the ASR peak.

### ADR-003: Silero VAD instead of NGC MarbleNet

`vad_multilingual_marblenet` is distributed through NGC and direct anonymous
download returned HTTP 403 during research. Silero VAD is MIT, language-agnostic,
bundled in the wheel, and plugs into NeMo through `external_vad_manifest`.
This removes an authenticated model source without weakening offline behavior.

### ADR-004: no LLM in the default path

The transcript is complete without an LLM. Meeting minutes are opt-in and talk
to a local OpenAI-compatible endpoint (llama.cpp server).
Long transcripts use a hierarchical map-reduce flow: `llm.chunk_chars` chunks are
summarised into structured JSON, then **aggregated deterministically** (near
duplicates merged; distinct decisions/tasks preserved). Themes are consolidated
to `llm.max_topics`, and each section is capped at
`llm.max_items_per_section`. The report uses the `asd-ste100` style by default:
short sentences, one idea per sentence, active voice, one term per concept. The
LLM is asked only for the final title and summary. Every `S<n>` reference is
validated against the real segment ids. Speaker names are user-provided; the LLM
may not invent them.

Speaker display names are an output-layer concern: `speaker_id` stays immutable
in JSON, names are applied by `label-speakers` or `--speaker-map`, and no ASR or
diarization stage has to be re-run.

### ADR-005: TOFU lock file for small-file integrity

Large LFS files have upstream SHA-256 metadata. Small JSON/tokenizer files do
not expose a checksum through the Hub API, so the downloader computes and stores
their hashes in `models/manifest.lock.json` on first download and verifies them
on every later run. Known hashes for all files are also pinned in
`models/manifest.yaml`.
