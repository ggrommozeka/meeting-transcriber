"""Diarization backend contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ..types import DiarizationInterval


@dataclass
class DiarizationResult:
    intervals: list[DiarizationInterval] = field(default_factory=list)
    backend: str = ""
    model: str = ""
    language_agnostic: bool = True
    failed: bool = False
    warnings: list[str] = field(default_factory=list)
    info: dict[str, object] = field(default_factory=dict)

    @property
    def speaker_ids(self) -> list[str]:
        seen: list[str] = []
        for interval in self.intervals:
            if interval.speaker_id not in seen:
                seen.append(interval.speaker_id)
        return seen


class Diarizer(ABC):
    """Abstract speaker diarization backend."""

    name: str = "abstract"

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def diarize(
        self,
        wav_path: str | Path,
        num_speakers: int | None = None,
    ) -> DiarizationResult:
        ...

    def release(self) -> None:  # noqa: B027
        """Drop model references and release accelerator memory."""
