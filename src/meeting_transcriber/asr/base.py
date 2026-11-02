"""ASR backend contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..audio.preprocess import AudioChunk
from ..types import Segment


@dataclass
class ASRResult:
    segments: list[Segment] = field(default_factory=list)
    language: str | None = None
    backend: str = ""
    model: str = ""
    words_available: bool = False
    warnings: list[str] = field(default_factory=list)
    info: dict[str, object] = field(default_factory=dict)


class Transcriber(ABC):
    """Abstract ASR backend.

    Implementations must never mutate the source audio and must return raw
    transcript text only. Terminology correction happens in a later stage.
    """

    name: str = "abstract"

    @abstractmethod
    def load(self) -> None:
        """Load model weights. Must fail clearly when weights are missing."""

    @abstractmethod
    def transcribe(
        self,
        chunks: list[AudioChunk],
        language: str = "ru",
        initial_prompt: str | None = None,
    ) -> ASRResult:
        ...

    def release(self) -> None:  # noqa: B027
        """Drop model references and release accelerator memory."""
