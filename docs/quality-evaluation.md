# Quality evaluation

Evaluation is a manual, reproducible workflow. No confidential recordings are
committed to the repository.

## Test set

Prepare `data/input/` with:

1. a mixed Russian-English technical meeting, 8–15 participants, 30–60 min;
2. a short clean recording with known speakers for smoke tests;
3. a recording with deliberate overlaps;
4. optionally, a long (> 2 h) recording for resume testing.

Store ground truth outside the repository or under `data/output/` (gitignored).

## ASR

Compute WER/CER with any external tool (e.g. `jiwer`) against a manually
verified reference transcript:

```bash
pip install jiwer
python - <<'PY'
import json, jiwer
data = json.load(open("data/output/meeting.json"))
hyp = " ".join(s["raw_text"] for s in data["segments"])
ref = open("reference.txt").read()
print("WER:", jiwer.wer(ref, hyp))
print("CER:", jiwer.cer(ref, hyp))
PY
```

Also measure separately:

* technical term accuracy (count correct canonical spellings);
* English terms embedded in Russian sentences;
* class/function/library identifiers;
* effect of `asr.initial_prompt`;
* effect of terminology post-correction (compare `raw_text` vs `corrected_text`).

Do not select a model solely by English leaderboard numbers.

## Diarization

DER requires reference RTTM. Use `nemo.collections.asr.metrics.der` or
`pyannote.metrics` externally:

```python
from nemo.collections.asr.metrics.der import score_labels
# score_labels(AUDIO_RTTM_MAP, reference, hypothesis, collar=0.25)
```

Report:

* DER, confusion, missed speech, false alarm;
* stability on short turns;
* behaviour with overlapping speech (this pipeline does not separate overlap);
* automatic speaker count vs. `--num-speakers`;

## End-to-end

Check:

* timestamps are monotonic and within the recording duration;
* every segment has a plausible speaker or an explicit quality flag;
* no text is lost at chunk boundaries (compare with a single-chunk run);
* JSON, Markdown, SRT agree;
* terminology corrections reference real source substrings;
* LLM report references only existing `S<n>` segment ids;
* speaker display names map to the intended `speaker_id` and do not alter ids.

## Suggested regression command

```bash
./scripts/offline_smoke_test.sh data/input/smoke.wav
pytest
```

Record results in `docs/known-limitations.md` or a separate evaluation log with
the exact model revisions and configuration fingerprint from the JSON output.
