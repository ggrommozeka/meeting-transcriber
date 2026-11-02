"""SRT subtitle writer."""

from __future__ import annotations

from pathlib import Path

from ..speakers import speaker_label, speaker_labels
from ..types import Transcript


def _srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def render_srt(transcript: Transcript) -> str:
    blocks: list[str] = []
    labels = speaker_labels(transcript)
    for index, segment in enumerate(transcript.segments, start=1):
        speaker = speaker_label(segment.speaker_id, labels)
        text = (segment.corrected_text or segment.raw_text).strip()
        if not text:
            continue
        blocks.append(
            f"{index}\n{_srt_time(segment.start)} --> {_srt_time(segment.end)}\n[{speaker}] {text}\n"
        )
    return "\n".join(blocks)


def write_srt(transcript: Transcript, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(render_srt(transcript), encoding="utf-8")
    tmp.replace(output)
    return output
