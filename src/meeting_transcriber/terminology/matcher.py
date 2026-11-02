"""Tokenisation, normalisation, exact/fuzzy/phonetic term matching."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from ..config import TerminologyConfig
from .glossary import Glossary, Term

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_]+")


@dataclass
class Token:
    text: str
    normalized: str
    start: int
    end: int


@dataclass
class Match:
    token_start: int
    token_end: int
    char_start: int
    char_end: int
    source: str
    term: Term
    confidence: float
    method: str


def normalize_token(text: str) -> str:
    return text.casefold().replace("ё", "е")


def tokenize(text: str) -> list[Token]:
    return [
        Token(text=m.group(0), normalized=normalize_token(m.group(0)), start=m.start(), end=m.end())
        for m in TOKEN_RE.finditer(text)
    ]


# Coarse phonetic skeleton, good enough to catch common ASR transliterations.
_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n",
    "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f",
    "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sh", "ъ": "", "ы": "i",
    "ь": "", "э": "e", "ю": "yu", "я": "ya",
}
_DIGRAPHS = [
    ("ph", "f"), ("ck", "k"), ("qu", "kv"), ("x", "ks"), ("th", "t"),
    ("sh", "s"), ("ch", "c"), ("zh", "z"), ("kh", "k"), ("ts", "c"),
]


def phonetic_key(text: str) -> str:
    lowered = text.casefold().replace("ё", "е")
    transliterated = "".join(_TRANSLIT.get(ch, ch) for ch in lowered)
    transliterated = re.sub(r"[^a-z0-9]", "", transliterated)
    for source, target in _DIGRAPHS:
        transliterated = transliterated.replace(source, target)
    # Remove vowels to make the key more tolerant to transliteration variants.
    skeleton = re.sub(r"[aeiouy]", "", transliterated)
    return skeleton or transliterated


def _similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _phonetic_eligible(source: str, alias: str) -> bool:
    """Phonetic matching is only a fallback for long, similar-length strings.

    Short tokens produce accidental skeleton collisions (for example ``втк`` and
    ``ветка`` collapse to the same consonant skeleton), which must never cause
    an automatic replacement. The consonant skeleton must also be long enough
    to carry information.
    """

    if len(source) < 5 or len(alias) < 5:
        return False
    if abs(len(source) - len(alias)) > 3:
        return False
    return len(phonetic_key(source)) >= 4


def find_matches(
    text: str,
    glossary: Glossary,
    cfg: TerminologyConfig,
) -> tuple[list[Match], list[str]]:
    """Return non-overlapping matches plus warnings about ambiguous candidates."""

    tokens = tokenize(text)
    if not tokens:
        return [], []

    entries: dict[int, list[tuple[list[str], Term]]] = {}
    for alias, term in glossary.alias_entries():
        alias_tokens = [t.normalized for t in tokenize(alias)]
        if not alias_tokens:
            continue
        entries.setdefault(len(alias_tokens), []).append((alias_tokens, term))

    candidates: list[Match] = []
    warnings: list[str] = []
    for start_index in range(len(tokens)):
        for size in range(1, min(cfg.max_ngram, len(tokens) - start_index) + 1):
            window = tokens[start_index : start_index + size]
            normalized = [t.normalized for t in window]
            joined = " ".join(normalized)
            best: Match | None = None
            second_confidence = 0.0

            for alias_tokens, term in entries.get(size, []):
                alias_joined = " ".join(alias_tokens)
                if normalized == alias_tokens:
                    confidence, method = 1.0, "alias"
                elif not cfg.fuzzy and not cfg.phonetic:
                    continue
                else:
                    source_norm = "".join(normalized)
                    alias_norm = "".join(alias_tokens)
                    if len(alias_norm) < 4:
                        continue
                    confidence = 0.0
                    method = ""
                    if cfg.fuzzy:
                        similarity = _similarity(source_norm, alias_norm)
                        if similarity >= max(0.72, cfg.threshold - 0.12):
                            confidence = similarity * 0.95
                            method = "fuzzy"
                    if cfg.phonetic and _phonetic_eligible(source_norm, alias_norm):
                        if phonetic_key(source_norm) == phonetic_key(alias_norm):
                            phonetic_confidence = 0.88
                            if phonetic_confidence > confidence:
                                confidence, method = phonetic_confidence, "phonetic"
                    if not method:
                        continue

                if term.case_sensitive and window[0].text != alias_tokens[0]:
                    # exact canonical spelling is still accepted case-sensitively
                    if joined != alias_joined:
                        continue

                confidence = min(confidence, 1.0)
                if confidence < cfg.threshold:
                    continue
                if term.canonical.casefold() == joined:
                    continue  # already correct, no correction needed

                if best is not None and term.term_id == best.term.term_id:
                    # Several aliases of the same term matching one span are not
                    # competing hypotheses.
                    if confidence > best.confidence:
                        best = Match(
                            token_start=start_index,
                            token_end=start_index + size - 1,
                            char_start=window[0].start,
                            char_end=window[-1].end,
                            source=text[window[0].start : window[-1].end],
                            term=term,
                            confidence=confidence,
                            method=method,
                        )
                    continue

                if best is None or confidence > best.confidence:
                    if best is not None:
                        second_confidence = max(second_confidence, best.confidence)
                    best = Match(
                        token_start=start_index,
                        token_end=start_index + size - 1,
                        char_start=window[0].start,
                        char_end=window[-1].end,
                        source=text[window[0].start : window[-1].end],
                        term=term,
                        confidence=confidence,
                        method=method,
                    )
                else:
                    second_confidence = max(second_confidence, confidence)

            if best is None:
                continue
            if best.confidence - second_confidence < cfg.ambiguous_margin and second_confidence > 0:
                warnings.append(
                    f"ambiguous term match skipped: '{best.source}' -> {best.term.canonical}"
                )
                continue
            candidates.append(best)

    # Resolve overlaps: prefer longer spans, then higher confidence.
    candidates.sort(key=lambda m: (m.token_start, -(m.token_end - m.token_start), -m.confidence))
    chosen: list[Match] = []
    for candidate in candidates:
        if chosen and candidate.token_start <= chosen[-1].token_end:
            continue
        chosen.append(candidate)
    return chosen, warnings


def apply_matches(text: str, matches: list[Match]) -> tuple[str, list[Match]]:
    """Return text with replacements applied and the matches that survived."""

    if not matches:
        return text, []
    pieces: list[str] = []
    cursor = 0
    applied: list[Match] = []
    for match in sorted(matches, key=lambda m: m.char_start):
        if match.char_start < cursor:
            continue
        pieces.append(text[cursor : match.char_start])
        pieces.append(match.term.canonical)
        cursor = match.char_end
        applied.append(match)
    pieces.append(text[cursor:])
    return "".join(pieces), applied
