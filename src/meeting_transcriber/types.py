"""Internal data contract shared by all pipeline stages.

The JSON emitted by :func:`Transcript.to_dict` is the canonical output format.
Every stage keeps its own data so results stay auditable: raw ASR text is never
mutated, corrections are recorded separately.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float | None = None
    speaker_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": _round(self.start),
            "end": _round(self.end),
            "word": self.word,
            "probability": _round(self.probability, 4),
            "speaker_id": self.speaker_id,
        }


@dataclass
class Correction:
    source: str
    target: str
    term_id: str
    confidence: float
    method: str = "alias"
    start: float | None = None
    end: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "term_id": self.term_id,
            "confidence": _round(self.confidence, 4),
            "method": self.method,
            "start": _round(self.start),
            "end": _round(self.end),
        }


@dataclass
class Segment:
    start: float
    end: float
    speaker_id: str | None
    raw_text: str
    corrected_text: str = ""
    words: list[Word] = field(default_factory=list)
    corrections: list[Correction] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)
    # Speaker ids whose diarization intervals overlap this segment.
    overlap_speakers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": _round(self.start),
            "end": _round(self.end),
            "speaker_id": self.speaker_id,
            "raw_text": self.raw_text,
            "corrected_text": self.corrected_text or self.raw_text,
            "words": [w.to_dict() for w in self.words],
            "corrections": [c.to_dict() for c in self.corrections],
            "quality_flags": list(self.quality_flags),
            "overlap_speakers": list(self.overlap_speakers),
        }


@dataclass
class Speaker:
    speaker_id: str
    display_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"speaker_id": self.speaker_id, "display_name": self.display_name}


@dataclass
class AudioMetadata:
    filename: str
    path: str | None = None
    duration: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    codec: str | None = None
    format: str | None = None
    bit_rate: int | None = None
    sha256: str | None = None
    has_video: bool = False
    video_codec: str | None = None
    audio_streams: int = 0
    audio_stream_index: int = 0
    extracted_audio: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["duration"] = _round(self.duration)
        return data


@dataclass
class DiarizationInterval:
    start: float
    end: float
    speaker_id: str
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": _round(self.start),
            "end": _round(self.end),
            "speaker_id": self.speaker_id,
            "confidence": _round(self.confidence, 4),
        }


@dataclass
class PipelineRunInfo:
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    tool_version: str | None = None
    git_commit: str | None = None
    device: str | None = None
    python_version: str | None = None
    vram_peak_mb: float | None = None
    stage_timings_seconds: dict[str, float] = field(default_factory=dict)
    library_versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class Transcript:
    meeting_id: str
    language: str
    source_audio: AudioMetadata
    segments: list[Segment] = field(default_factory=list)
    speakers: list[Speaker] = field(default_factory=list)
    models: dict[str, Any] = field(default_factory=dict)
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    run: PipelineRunInfo = field(default_factory=PipelineRunInfo)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    diarization_intervals: list[DiarizationInterval] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "meeting_id": self.meeting_id,
            "language": self.language,
            "source_audio": self.source_audio.to_dict(),
            "speakers": [s.to_dict() for s in self.speakers],
            "segments": [s.to_dict() for s in self.segments],
            "diarization_intervals": [d.to_dict() for d in self.diarization_intervals],
            "models": self.models,
            "config": self.config_snapshot,
            "run": self.run.to_dict(),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "quality_flags": list(self.quality_flags),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @property
    def speaker_ids(self) -> list[str]:
        seen: list[str] = []
        for segment in self.segments:
            if segment.speaker_id and segment.speaker_id not in seen:
                seen.append(segment.speaker_id)
        return seen
