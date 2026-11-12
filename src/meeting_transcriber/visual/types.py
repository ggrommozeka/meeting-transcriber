"""Data contract for video frame OCR results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0"


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)


@dataclass
class CodeCard:
    """A stable screen state with its local OCR text.

    ``frame`` is a file name (not an absolute path) so report links stay
    portable when the frames are copied next to the Markdown output.
    """

    code_id: str
    start: float
    end: float
    kind: str
    text: str
    code_lines: list[str] = field(default_factory=list)
    frame: str | None = None
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_id": self.code_id,
            "start": _round(self.start),
            "end": _round(self.end),
            "kind": self.kind,
            "text": self.text,
            "code_lines": list(self.code_lines),
            "frame": self.frame,
            "confidence": _round(self.confidence, 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodeCard:
        return cls(
            code_id=str(data.get("code_id") or "C?"),
            start=float(data.get("start") or 0.0),
            end=float(data.get("end") or 0.0),
            kind=str(data.get("kind") or "slide"),
            text=str(data.get("text") or ""),
            code_lines=[str(line) for line in data.get("code_lines") or []],
            frame=str(data["frame"]) if data.get("frame") else None,
            confidence=(
                float(data["confidence"]) if data.get("confidence") is not None else None
            ),
        )


@dataclass
class VisualResult:
    cards: list[CodeCard] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SCHEMA_VERSION,
            "cards": [card.to_dict() for card in self.cards],
            "stats": dict(self.stats),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VisualResult:
        return cls(
            cards=[CodeCard.from_dict(item) for item in data.get("cards") or []],
            stats=dict(data.get("stats") or {}),
            warnings=[str(item) for item in data.get("warnings") or []],
        )

    @property
    def code_cards(self) -> list[CodeCard]:
        return [card for card in self.cards if card.kind == "code"]
