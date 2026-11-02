"""Markdown transcript and meeting minutes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..postprocessing.meeting_report import render_report_markdown
from ..speakers import speaker_label, speaker_labels
from ..types import Transcript


def _timestamp(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def render_markdown(transcript: Transcript, report: dict[str, Any] | None = None) -> str:
    meta = transcript.source_audio
    lines = [
        f"# Транскрипция: {transcript.meeting_id}",
        "",
        f"- Язык: `{transcript.language}`",
        f"- Файл: `{meta.filename}`",
        f"- Длительность: {_timestamp(meta.duration or 0)}",
        f"- Модель ASR: `{transcript.models.get('asr', 'unknown')}`",
        f"- Модель diarization: `{transcript.models.get('diarization', 'unknown')}`",
        "",
    ]
    if transcript.warnings:
        lines += ["## Предупреждения", ""]
        lines += [f"- {warning}" for warning in transcript.warnings]
        lines.append("")

    lines += ["## Расшифровка", ""]
    labels = speaker_labels(transcript)
    for index, segment in enumerate(transcript.segments, start=1):
        speaker = speaker_label(segment.speaker_id, labels)
        if segment.speaker_id and segment.speaker_id in labels:
            speaker = f"{speaker} ({segment.speaker_id})"
        text = segment.corrected_text or segment.raw_text
        flags = f" _(flags: {', '.join(segment.quality_flags)})_" if segment.quality_flags else ""
        lines.append(
            f"**[S{index}] {_timestamp(segment.start)} {speaker}:** {text}{flags}"
        )
        lines.append("")

    if transcript.segments and any(s.corrections for s in transcript.segments):
        lines += ["## Терминологические исправления", ""]
        for index, segment in enumerate(transcript.segments, start=1):
            for correction in segment.corrections:
                lines.append(
                    f"- [S{index}] `{correction.source}` → `{correction.target}` "
                    f"({correction.method}, confidence={correction.confidence:.2f})"
                )
        lines.append("")

    if report:
        lines += ["---", "", render_report_markdown(report)]

    return "\n".join(lines).rstrip() + "\n"


def write_markdown(
    transcript: Transcript,
    path: str | Path,
    report: dict[str, Any] | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(render_markdown(transcript, report), encoding="utf-8")
    tmp.replace(output)
    return output
