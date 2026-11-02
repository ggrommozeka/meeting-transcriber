"""Canonical JSON writer."""

from __future__ import annotations

import json
from pathlib import Path

from ..types import Transcript


def transcript_to_json(transcript: Transcript, include_words: bool = True) -> str:
    data = transcript.to_dict()
    if not include_words:
        for segment in data["segments"]:
            segment.pop("words", None)
    return json.dumps(data, ensure_ascii=False, indent=2)


def write_json(transcript: Transcript, path: str | Path, include_words: bool = True) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(transcript_to_json(transcript, include_words), encoding="utf-8")
    tmp.replace(output)
    return output
