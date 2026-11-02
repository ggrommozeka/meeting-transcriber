# Known limitations

Everything below is either a deliberate scope decision or an unverified
assumption. Nothing here should be read as a measured guarantee.

## Unverified in this repository

* **WER/DER and speaker-count accuracy** were not measured against ground truth.
  One real 87-minute Russian meeting produced 8 speakers, 945 segments, no
  unlabelled segments, and 4612 MiB peak VRAM on an RTX 4060; this is a sanity
  check, not a benchmark. Measure DER with reference RTTM before relying on
  speaker labels for attribution.
* **TitaNet-Large on Russian meetings** is not validated against ground truth.
  It is an English-trained speaker-embedding model widely used for multilingual
  diarization; the observed clustering was plausible, but Russian-specific DER
  is unknown.
* **Automatic speaker counting** was tested on one long recording (8 speakers).
  Accuracy on 8–15 participant meetings is not measured. Provide
  `--num-speakers` when the exact count is known.
* **Docker image is built and tested** (8.14 GB): full 87-minute ASR +
  diarization in the container produced 918 segments and 8 speakers in 2m40s
  with 4612 MiB peak VRAM; a 60-second clip exercised the same path. The LLM
  report is intentionally disabled in the container because Compose uses
  `network_mode: "none"`; run it natively. Reproducibility still depends on the
  mutable `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` tag.
* **NeMo version compatibility**: `ClusteringDiarizer` exists in NeMo 2.6.2
  (source inspected and exercised), but NeMo 3.x may change or remove it. The
  dependency is pinned for that reason.
* **Long-recording diarization uses single-scale embeddings** above 30 minutes.
  This was necessary to avoid an O(N²) clustering OOM on 8 GB. Temporal
  resolution is therefore coarser than for short multi-scale runs.

## Architectural limitations

* **Overlapping speech is not separated.** Diarization reports multiple active
  speakers (`overlap_speakers`) but assigns one label per word. True
  overlap-aware diarization would need a different backend.
* **Speaker ids are not names.** Mapping `speaker_00` to a person is a manual
  step; the project intentionally has no voice-print database.
* **Chunk merging uses the overlap midpoint.** It is designed to avoid both
  duplication and loss, but a word whose midpoint falls just on the wrong side
  of a boundary can still be dropped. Reduce `overlap_seconds` issues by
  increasing overlap (default 15 s).
* **No aggressive denoising.** Preprocessing only resamples and downmixes.
  Recordings with heavy noise or clipping will degrade both ASR and diarization.
* **NeMo writes intermediate files** in the meeting work directory. They are
  pruned by default (`diarization.keep_intermediate: false`), but embeddings and
  VAD output may temporarily contain voice-derived data.
* **Small model files are trust-on-first-use.** Their SHA-256 is recorded in
  `models/manifest.lock.json` on first download because the Hub API does not
  expose a checksum for them. All files also have pinned hashes in
  `models/manifest.yaml` where upstream metadata provides them.
* **LLM protocol quality is not benchmarked.** Chunk extraction can miss
  decisions or tasks, and the summary is model-generated. The reduce phase is
  deterministic to avoid losing distinct items, but verify the `S<n>` references
  and treat the report as a draft. The tested setup is unsloth Qwen3.5-9B
  UD-Q4_K_XL via llama.cpp; other OpenAI-compatible servers should work but are
  not tested here.
* **Speaker display names are manual.** The pipeline never infers a person's
  identity. The template shows sample quotes; mapping `speaker_XX` to a real
  name is a human decision.
* **Terminology correction is conservative by default.** Phonetic matching is
  disabled (`terminology.phonetic: false`), and ambiguous candidates are skipped.
  A broad phonetic matcher produced false positives on ordinary Russian words
  ("куда" → CUDA, "удержаться" → diarization) during evaluation; enable it only
  for a curated project glossary. Even fuzzy matching can still be wrong; check
  the `corrections` array.

## Deliberate exclusions

* No web interface, database, queue, or concurrent job scheduler.
* No cloud LLM or cloud ASR.
* No automatic model download at runtime.
* No HF token in runtime configuration.
* No built-in speaker-name assignment.
