"""Visual stage orchestration: frames -> OCR -> classified code cards."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from ..config import VisualConfig
from ..errors import AudioError, TranscriberError
from ..visual.frames import (
    FrameGroup,
    extract_samples,
    group_samples,
    save_code_frame,
)
from ..visual.ocr import TesseractOCR
from ..visual.types import CodeCard, VisualResult

_CODE_KEYWORDS = re.compile(
    r"\b(?:class|struct|enum|union|namespace|template|typename|typedef|using|"
    r"def|function|return|if|else|elif|for|while|switch|case|break|continue|"
    r"public|private|protected|virtual|override|const|constexpr|static|inline|"
    r"void|bool|char|short|int|long|float|double|auto|size_t|std|new|delete|"
    r"include|import|from|package|lambda|async|await|yield|try|catch|throw)\b",
    re.IGNORECASE,
)
_CODE_LINE = re.compile(
    r"(?:[{};]\s*$|"
    r"[A-Za-z_]\w*(?:::\w+)*(?:<[^>]*>)?\s*\([^)]*\)\s*(?:const\s*)?[{;:]|"
    r"#\s*(?:include|define|if|ifdef|endif|pragma)\b|"
    r"(?:->|=>|::|==|!=|<=|>=|&&|\|\||\+=|-=|\*=|/=)\s*)"
)
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_UI_NOISE = re.compile(
    r"(?:^|\b)(?:zoom|mute|unmute|camera|screen share|chat|participants|"
    r"record|recording|you are viewing|stop sharing|present now|"
    r"explorer|outline|timeline|problems|terminal|debug console|ports|"
    r"демонстрация экрана|выключить звук|участники|чат)(?:\b|$)",
    re.IGNORECASE,
)
_COMMENT_LINE = re.compile(r"^\s*(?://|/\*|\*|\*/)")
_SOURCE_FILE = re.compile(r"\.(?:hpp|hxx|cpp|cxx|cc|h)\b", re.IGNORECASE)


def _code_lines(lines: list[str]) -> list[str]:
    result: list[str] = []
    for line in lines:
        stripped = line.rstrip()
        if not stripped.strip():
            continue
        if _UI_NOISE.search(stripped):
            continue
        # Tab strips and file trees survive cropping; they contain several
        # source file names on one line and are not code.
        if stripped.count("©") >= 2 or len(_SOURCE_FILE.findall(stripped)) >= 2:
            continue
        if _COMMENT_LINE.match(stripped) or _looks_like_code(stripped):
            result.append(stripped)
    return result


def _looks_like_code(line: str) -> bool:
    if _CYRILLIC.search(line) and not _CODE_LINE.search(line):
        return False
    compact = line.replace(" ", "")
    if compact:
        ascii_alnum = sum(1 for char in compact if char.isascii() and char.isalnum())
        if ascii_alnum / len(compact) < 0.3:
            return False
    return bool(_CODE_KEYWORDS.search(line) or _CODE_LINE.search(line))


def classify_text(text: str, min_chars: int) -> str:
    """Return ``code``, ``slide`` or ``noise`` for one OCR result."""

    raw_lines = [line for line in text.splitlines() if line.strip()]
    total_chars = sum(len(line.strip()) for line in raw_lines)
    if total_chars < min_chars or not raw_lines:
        return "noise"

    code_hits = sum(1 for line in raw_lines if _looks_like_code(line))
    braces = text.count("{") + text.count("}") + text.count(";")
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    ascii_ratio = ascii_chars / max(1, len(text))
    code_score = code_hits / len(raw_lines)
    cyrillic_ratio = len(_CYRILLIC.findall(text)) / max(1, len(text))
    if cyrillic_ratio > 0.35 and code_score < 0.4:
        return "slide"
    if code_score >= 0.25 or (code_hits >= 2 and braces >= 2 and ascii_ratio >= 0.45):
        return "code"
    return "slide"


def _clean_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\f", "\n").splitlines()]
    cleaned: list[str] = []
    blank = False
    for line in lines:
        if not line.strip():
            if cleaned and not blank:
                cleaned.append("")
            blank = True
            continue
        cleaned.append(line)
        blank = False
    return "\n".join(cleaned).strip()


def _pick_groups(groups: list[FrameGroup], limit: int) -> list[FrameGroup]:
    if len(groups) <= limit:
        return groups
    step = max(1, len(groups) // limit)
    return groups[::step][:limit]


def extract_visual(
    source: str | Path,
    work_dir: str | Path,
    cfg: VisualConfig,
    ffmpeg: str,
    duration: float,
) -> VisualResult:
    """Run the full visual stage. Never raises for expected OCR problems."""

    src = Path(source).expanduser().resolve()
    work = Path(work_dir).expanduser().resolve()
    results = VisualResult(enabled=True)
    ocr = TesseractOCR(cfg)
    if not ocr.available:
        results.warnings.append(
            "visual stage skipped: tesseract executable not found; install "
            "tesseract-ocr (rus+eng) or set visual.enabled=false"
        )
        results.enabled = False
        return results

    frames_dir = work / "visual_frames"
    cards_dir = work / "frames"
    if cards_dir.is_dir():
        shutil.rmtree(cards_dir)
    cards_dir.mkdir(parents=True, exist_ok=True)

    try:
        samples = extract_samples(ffmpeg, src, work, cfg)
        results.stats["frames_sampled"] = len(samples)
        groups = group_samples(samples, duration, cfg.dedupe_distance)
        results.stats["screen_states"] = len(groups)
        if len(groups) > cfg.max_cards:
            results.warnings.append(
                f"visual stage capped screen states to {cfg.max_cards}"
            )
        groups = _pick_groups(groups, cfg.max_cards)

        cards: list[CodeCard] = []
        ocr_errors = 0
        for group in groups:
            text, warning = ocr.recognize(group.sample.path)
            if warning:
                ocr_errors += 1
                if ocr_errors <= 3:
                    results.warnings.append(warning)
                continue
            text = _clean_text(text)
            kind = classify_text(text, cfg.min_text_chars)
            if kind == "noise":
                continue
            code_lines = _code_lines([line for line in text.splitlines() if line.strip()])
            if kind == "code" and code_lines:
                text = "\n".join(code_lines)
            code_id = f"C{len(cards) + 1}"
            frame = None
            if kind == "code" and cfg.keep_code_frames:
                frame = save_code_frame(group, cards_dir, code_id)
            cards.append(
                CodeCard(
                    code_id=code_id,
                    start=round(group.start, 3),
                    end=round(group.end, 3),
                    kind=kind,
                    text=text,
                    code_lines=code_lines,
                    frame=frame,
                )
            )
            if len(cards) >= cfg.max_cards:
                break

        results.cards = cards
        results.stats["cards"] = len(cards)
        results.stats["code_cards"] = sum(1 for card in cards if card.kind == "code")
        if ocr_errors:
            results.stats["ocr_errors"] = ocr_errors
            results.warnings.append(f"visual stage: {ocr_errors} frame(s) failed OCR")
        if not cards:
            results.warnings.append("visual stage: no readable screen content found")
    except (AudioError, TranscriberError) as exc:
        results.warnings.append(f"visual stage failed: {exc}")
    finally:
        if frames_dir.is_dir():
            shutil.rmtree(frames_dir, ignore_errors=True)

    return results


def load_visual_result(work_dir: str | Path) -> VisualResult | None:
    path = Path(work_dir).expanduser().resolve() / "visual.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return VisualResult.from_dict(data)


__all__ = [
    "VisualResult",
    "classify_text",
    "extract_visual",
    "load_visual_result",
]
