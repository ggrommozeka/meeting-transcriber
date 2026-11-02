"""Terminology correction with a full audit trail."""

from __future__ import annotations

from pathlib import Path

from ..config import TerminologyConfig
from ..types import Correction, Segment
from .glossary import Glossary, load_glossaries
from .matcher import apply_matches, find_matches, tokenize


class TerminologyCorrector:
    def __init__(self, glossary: Glossary, cfg: TerminologyConfig) -> None:
        self.glossary = glossary
        self.cfg = cfg

    @classmethod
    def from_config(
        cls,
        cfg: TerminologyConfig,
        terminology_dir: str | Path,
        extra_files: list[str] | None = None,
    ) -> TerminologyCorrector:
        glossary = load_glossaries(cfg.profiles, terminology_dir, extra_files)
        return cls(glossary, cfg)

    def correct_segment(self, segment: Segment) -> list[str]:
        """Populate ``corrected_text`` and ``corrections``; return warnings."""

        matches, warnings = find_matches(segment.raw_text, self.glossary, self.cfg)
        corrected, applied = apply_matches(segment.raw_text, matches)
        if not self.cfg.apply_corrections:
            corrected = segment.raw_text

        token_count = len(tokenize(segment.raw_text))
        corrections: list[Correction] = []
        for match in applied:
            start: float | None = None
            end: float | None = None
            if segment.words and token_count == len(segment.words):
                start = segment.words[match.token_start].start
                end = segment.words[match.token_end].end
            corrections.append(
                Correction(
                    source=match.source,
                    target=match.term.canonical,
                    term_id=match.term.term_id,
                    confidence=match.confidence,
                    method=match.method,
                    start=start,
                    end=end,
                )
            )

        segment.corrected_text = corrected
        segment.corrections = corrections
        return warnings

    def correct_segments(self, segments: list[Segment]) -> list[str]:
        warnings: list[str] = []
        for segment in segments:
            warnings.extend(self.correct_segment(segment))
        return warnings
