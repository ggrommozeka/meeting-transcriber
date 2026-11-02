# Model selection

Research was performed with the Hugging Face MCP (model cards, revisions,
licenses, file metadata, SHA-256) and DeepWiki MCP (faster-whisper and NeMo
APIs). No model name or API argument in this repository is taken from memory.

## Requirements

* Russian speech with frequent English technical terms
* 8–15 participants, no fixed 4-speaker cap
* word timestamps where available
* 8 GB VRAM, sequential loading
* fully local runtime, no HF token

## Selected stack

| Stage | Selection | Why |
| --- | --- | --- |
| ASR | `deepdml/faster-whisper-large-v3-turbo-ct2` @ `4df90f7` | multilingual Whisper large-v3-turbo, CTranslate2 format, MIT, 1.5 GiB, fast on 8 GB |
| VAD | Silero VAD 6.2.3 | language-agnostic, MIT, bundled model, no NGC auth |
| Speaker embeddings | `nvidia/speakerverification_en_titanet_large` @ `0dc382f` | official NeMo `.nemo`, CC-BY-4.0, 97 MiB, standard component of NeMo diarization |
| Clustering | NeMo spectral clustering (`ClusteringDiarizer`) | automatic speaker count with `max_num_speakers=20` |
| Post-processing | unsloth Qwen3.5-9B UD-Q4_K_XL via llama.cpp | optional local protocol, no cloud dependency |

## Report model selection

The report stage was the weakest part of the first end-to-end run. Evaluation
with HF MCP produced these candidates:

| Model | Params | RU support | Context | Q4 size | Decision |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-9B (unsloth) | 9.65B | 201 languages | 262k | 5.56 GiB UD-Q4_K_XL | **selected** |
| Qwen3-8B | 8.19B | 100+ languages | 32k | 4.68 GiB | fallback |
| Ministral-3-8B | 8.4B | Russian not listed | — | 4.84 GiB | rejected |
| Vikhr-Nemo-12B | 12.2B | ru/en | 128k | ~7.3 GiB | rejected: does not fit 8 GB with context |
| Gemma-3-12B | 12.2B | ru | 128k | ~7.5 GiB | rejected: gated and too large |
| T-lite-instruct-0.1 | 8.03B | ru-native | 8k | community GGUF | not used: 2024 model, weaker JSON/instruction following |

Qwen3.5-9B is served by llama.cpp with thinking disabled. This matters: with the
default chat template the model spends the whole token budget in
`reasoning_content` and returns empty `content`. `--reasoning off` fixes that.

Prompt/architecture changes were required in addition to the model:

* topic consolidation from 29 granular items to at most 6 themes;
* strict definition of a decision (explicit agreement, not discussion);
* strict section limits: at most 6 items per section and a 5-sentence summary;
* ASD-STE100-based style rules: short sentences, one idea per sentence, active
  voice, consistent terminology, no marketing language;
* context-sized grouping for consolidation so the prompt fits 8k;
* deterministic aggregation, so the LLM cannot drop decisions/tasks during a
  merge;
* speaker-id scrubbing from generated text.

## Evidence checked during research

### faster-whisper

* `WhisperModel(model_size_or_path=<local dir>, device=..., compute_type=...,
  local_files_only=True)` loads CTranslate2 weights without a Hub lookup.
* `transcribe()` supports `language`, `word_timestamps`, `vad_filter`,
  `initial_prompt`, `condition_on_previous_text`, `beam_size`.
* Segments expose `start`, `end`, `text`, `avg_logprob`, `no_speech_prob`,
  `compression_ratio`; words expose `start`, `end`, `word`, `probability`.

### NeMo

* `ClusteringDiarizer` remains available in NeMo 2.x at
  `nemo.collections.asr.models.ClusteringDiarizer`.
* Config keys used here: `diarizer.manifest_filepath`, `out_dir`,
  `oracle_vad`, `vad.model_path`, `vad.external_vad_manifest`,
  `speaker_embeddings.model_path`, `clustering.parameters.oracle_num_speakers`,
  `clustering.parameters.max_num_speakers`.
* `max_num_speakers: 20` is supported by spectral clustering; there is no
  four-speaker hard limit.
* External VAD manifests are line-delimited JSON with `audio_filepath`,
  `offset`, `duration`, `uniq_id`; this is what Silero feeds.
* Results are written to `<out_dir>/pred_rttms/<uniq_id>.rttm`.

### Rejected alternatives

| Candidate | Reason |
| --- | --- |
| pyannote.audio pipelines | model downloads are gated and require a HF token; conflicts with the no-token runtime rule |
| Sortformer / end-to-end diarizers | fixed speaker limits, typically 4; unsuitable for 8–15 |
| NGC `vad_multilingual_marblenet` | anonymous direct download returned HTTP 403; would add NGC authentication |
| Qwen3-ASR | Russian quality, license, VRAM, and inference maturity were not confirmed; left as a future comparison behind `Transcriber` |
| Whisper large-v3 | higher accuracy but ~3 GiB and slower; turbo chosen for the 8 GB target |

## Version pins

| Dependency | Pin |
| --- | --- |
| Python | >= 3.10 (Docker image uses 3.11) |
| torch | 2.6.0 (Docker base `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`) |
| nemo_toolkit[asr] | 2.6.2 |
| faster-whisper | 1.1.1 |
| silero-vad | 6.2.3 |

NeMo 2.6.2 declares `torch>=2.6.0`; the base image satisfies it exactly.
`ClusteringDiarizer` was verified in the v2.6.2 source tree during research.

## What was not verified

The reference 87-minute meeting produced 8 speakers and 4612 MiB peak VRAM on an
RTX 4060, but the following are engineering expectations, not benchmarks:

* DER/WER on Russian 8–15 participant recordings;
* accuracy of automatic speaker counting on other meetings;
* TitaNet-Large behaviour across different recording setups.

See `docs/known-limitations.md` and `docs/gpu-memory.md` for the measurement
procedure.
