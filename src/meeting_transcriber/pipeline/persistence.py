"""Serialisation helpers for resumable pipeline stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..asr.base import ASRResult
from ..diarization.base import DiarizationResult
from ..types import (
    AudioMetadata,
    Correction,
    DiarizationInterval,
    PipelineRunInfo,
    Segment,
    Speaker,
    Transcript,
    Word,
)


def word_to_dict(word: Word) -> dict[str, Any]:
    return word.to_dict()


def _word_from_dict(data: dict[str, Any]) -> Word:
    return Word(
        start=float(data["start"]),
        end=float(data["end"]),
        word=str(data.get("word", "")),
        probability=data.get("probability"),
        speaker_id=data.get("speaker_id"),
    )


def _correction_from_dict(data: dict[str, Any]) -> Correction:
    return Correction(
        source=str(data.get("source", "")),
        target=str(data.get("target", "")),
        term_id=str(data.get("term_id", "")),
        confidence=float(data.get("confidence", 0.0)),
        method=str(data.get("method", "alias")),
        start=data.get("start"),
        end=data.get("end"),
    )


def segment_from_dict(data: dict[str, Any]) -> Segment:
    return Segment(
        start=float(data["start"]),
        end=float(data["end"]),
        speaker_id=data.get("speaker_id"),
        raw_text=str(data.get("raw_text", "")),
        corrected_text=str(data.get("corrected_text", "")),
        words=[_word_from_dict(w) for w in data.get("words", [])],
        corrections=[_correction_from_dict(c) for c in data.get("corrections", [])],
        quality_flags=list(data.get("quality_flags", [])),
        overlap_speakers=list(data.get("overlap_speakers", [])),
    )


def audio_metadata_from_dict(data: dict[str, Any]) -> AudioMetadata:
    allowed = AudioMetadata.__dataclass_fields__.keys()
    return AudioMetadata(**{k: v for k, v in data.items() if k in allowed})


def transcript_from_dict(data: dict[str, Any]) -> Transcript:
    run_data = data.get("run") or {}
    run = PipelineRunInfo(
        **{k: v for k, v in run_data.items() if k in PipelineRunInfo.__dataclass_fields__}
    )
    return Transcript(
        meeting_id=str(data["meeting_id"]),
        language=str(data.get("language", "ru")),
        source_audio=audio_metadata_from_dict(data["source_audio"]),
        segments=[segment_from_dict(s) for s in data.get("segments", [])],
        speakers=[Speaker(**s) for s in data.get("speakers", [])],
        models=dict(data.get("models") or {}),
        config_snapshot=dict(data.get("config") or {}),
        run=run,
        warnings=list(data.get("warnings") or []),
        errors=list(data.get("errors") or []),
        diarization_intervals=[
            DiarizationInterval(
                start=float(item["start"]),
                end=float(item["end"]),
                speaker_id=str(item["speaker_id"]),
                confidence=item.get("confidence"),
            )
            for item in data.get("diarization_intervals", [])
        ],
        quality_flags=list(data.get("quality_flags") or []),
    )


def save_transcript(transcript: Transcript, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(transcript.to_json(), encoding="utf-8")
    tmp.replace(path)


def load_transcript(path: Path) -> Transcript:
    return transcript_from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_asr_result(result: ASRResult, path: Path) -> None:
    payload = {
        "language": result.language,
        "backend": result.backend,
        "model": result.model,
        "words_available": result.words_available,
        "warnings": result.warnings,
        "info": result.info,
        "segments": [s.to_dict() for s in result.segments],
    }
    _write_json(path, payload)


def load_asr_result(path: Path) -> ASRResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ASRResult(
        segments=[segment_from_dict(s) for s in data.get("segments", [])],
        language=data.get("language"),
        backend=str(data.get("backend", "")),
        model=str(data.get("model", "")),
        words_available=bool(data.get("words_available")),
        warnings=list(data.get("warnings") or []),
        info=dict(data.get("info") or {}),
    )


def save_diarization_result(result: DiarizationResult, path: Path) -> None:
    payload = {
        "backend": result.backend,
        "model": result.model,
        "failed": result.failed,
        "warnings": result.warnings,
        "info": result.info,
        "intervals": [interval.to_dict() for interval in result.intervals],
    }
    _write_json(path, payload)


def load_diarization_result(path: Path) -> DiarizationResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    return DiarizationResult(
        intervals=[
            DiarizationInterval(
                start=float(item["start"]),
                end=float(item["end"]),
                speaker_id=str(item["speaker_id"]),
                confidence=item.get("confidence"),
            )
            for item in data.get("intervals", [])
        ],
        backend=str(data.get("backend", "")),
        model=str(data.get("model", "")),
        failed=bool(data.get("failed", False)),
        warnings=list(data.get("warnings") or []),
        info=dict(data.get("info") or {}),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
