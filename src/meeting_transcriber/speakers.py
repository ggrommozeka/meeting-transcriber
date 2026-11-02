"""Speaker naming: review templates, manual maps, display-name application.

Speaker ids stay immutable in the JSON contract. Display names are cosmetic and
always user-provided; the pipeline never guesses a person's identity from voice
or text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError
from .types import Segment, Transcript


@dataclass
class SpeakerStats:
    speaker_id: str
    talk_time_seconds: float = 0.0
    segments: int = 0
    samples: list[str] = field(default_factory=list)

    @property
    def talk_time_minutes(self) -> float:
        return round(self.talk_time_seconds / 60, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": "",
            "talk_time_seconds": round(self.talk_time_seconds, 1),
            "segments": self.segments,
            "samples": self.samples,
        }


def collect_speaker_stats(
    transcript: Transcript,
    sample_count: int = 3,
    sample_max_chars: int = 220,
    sample_min_duration: float = 3.0,
) -> list[SpeakerStats]:
    stats: dict[str, SpeakerStats] = {}
    for segment in transcript.segments:
        speaker = segment.speaker_id
        if not speaker:
            continue
        entry = stats.setdefault(speaker, SpeakerStats(speaker_id=speaker))
        entry.talk_time_seconds += max(0.0, segment.end - segment.start)
        entry.segments += 1

    for segment in transcript.segments:
        speaker = segment.speaker_id
        if not speaker or segment.end - segment.start < sample_min_duration:
            continue
        entry = stats[speaker]
        text = (segment.corrected_text or segment.raw_text).strip()
        if len(text) < 20:
            continue
        if text in entry.samples:
            continue
        if len(entry.samples) < sample_count:
            entry.samples.append(text[:sample_max_chars])

    return sorted(stats.values(), key=lambda s: s.talk_time_seconds, reverse=True)


def speaker_template(
    transcript: Transcript, stats: list[SpeakerStats] | None = None
) -> dict[str, Any]:
    stats = stats or collect_speaker_stats(transcript)
    return {
        "meeting_id": transcript.meeting_id,
        "speakers": {entry.speaker_id: entry.to_dict() for entry in stats},
    }


def write_speaker_template(
    transcript: Transcript, path: str | Path, overwrite: bool = False
) -> Path:
    output = Path(path).expanduser()
    if output.exists() and not overwrite:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(
            speaker_template(transcript), allow_unicode=True, sort_keys=False
        ),
        encoding="utf-8",
    )
    return output


def load_speaker_map(path: str | Path) -> dict[str, str]:
    """Load ``speaker_00: Иван`` names from YAML.

    Supports both the generated template (``speakers: {id: {name: ...}}``) and a
    flat mapping (``speakers: {id: name}`` or top-level ``id: name``).
    """

    source = Path(path).expanduser()
    if not source.is_file():
        raise ConfigError(f"speaker map not found: {source}")
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid speaker map YAML in {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"{source}: expected a mapping")

    raw = payload.get("speakers", payload)
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: 'speakers' must be a mapping")

    mapping: dict[str, str] = {}
    for speaker_id, value in raw.items():
        if isinstance(value, dict):
            name = value.get("name")
        else:
            name = value
        if name is None:
            continue
        name = str(name).strip()
        if name:
            mapping[str(speaker_id)] = name
    return mapping


def apply_speaker_names(transcript: Transcript, mapping: dict[str, str]) -> int:
    """Set ``display_name`` on known speakers; return number of names applied."""

    applied = 0
    for speaker in transcript.speakers:
        name = mapping.get(speaker.speaker_id)
        if name:
            speaker.display_name = name
            applied += 1
    return applied


def speaker_labels(transcript: Transcript) -> dict[str, str]:
    return {
        speaker.speaker_id: speaker.display_name
        for speaker in transcript.speakers
        if speaker.display_name
    }


def speaker_label(speaker_id: str | None, labels: dict[str, str]) -> str:
    if not speaker_id:
        return "unknown"
    return labels.get(speaker_id, speaker_id)


def segment_label(segment: Segment, labels: dict[str, str]) -> str:
    return speaker_label(segment.speaker_id, labels)
