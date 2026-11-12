"""Meeting minutes generation through a local OpenAI-compatible LLM.

Long transcripts are processed hierarchically (map-reduce): each chunk is
summarised into a structured JSON fragment, then fragments are merged. Every
item keeps references to original ``S<n>`` segments, and unknown references are
dropped. The LLM may not invent speakers, decisions, owners or deadlines.

The stage is optional and never blocks transcript output.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ..config import LLMConfig
from ..errors import PipelineError
from ..speakers import collect_speaker_stats, speaker_label, speaker_labels
from ..types import Segment, Transcript
from ..visual.context import build_visual_context
from ..visual.types import CodeCard
from .llm_client import LLMClient

SYSTEM_PROMPT = """Ты — ассистент протоколирования совещаний. Тебе дан фрагмент транскрипта с идентификаторами сегментов.
Строгие правила:
1. Используй только факты из транскрипта. Не придумывай реплики, имена, решения, сроки и ответственных.
2. Каждый вывод обязан ссылаться на сегменты через их идентификаторы (например, S12), и только на те, что есть во фрагменте.
3. "decisions" — только явно согласованные решения или выбор. Не записывай сюда предложения, идеи и общее обсуждение.
4. "action_items" — только конкретные следующие действия. owner и due указывай, только если они явно названы, иначе null.
5. "topics" — 2–4 крупные темы фрагмента, а не по одной теме на реплику.
6. Если данных недостаточно, укажи это в "uncertain" и поставь confidence ниже 0.6.
7. Не переписывай речи и не меняй их смысл.
8. Если в запросе есть блок «На экране», это вспомогательный OCR-контекст. Используй его только для точных имён классов, функций, файлов и терминов. Не выдумывай содержимое экрана.
9. Поле "code_refs" заполняй идентификаторами карточек (C1, C2, ...), только если утверждение связано с показанным на экране. Идентификаторы бери только из блока «На экране».
10. Верни только валидный JSON без markdown-ограждений.

Схема ответа:
{
  "topics": [{"title": "строка", "summary": "строка", "segment_refs": ["S1"], "code_refs": ["C1"]}],
  "decisions": [{"text": "строка", "segment_refs": ["S1"], "code_refs": ["C1"], "confidence": 0.0}],
  "action_items": [{"task": "строка", "owner": null, "due": null, "segment_refs": ["S1"], "code_refs": ["C1"], "confidence": 0.0}],
  "risks": [{"text": "строка", "segment_refs": ["S1"], "code_refs": ["C1"]}],
  "open_questions": [{"text": "строка", "segment_refs": ["S1"], "code_refs": ["C1"]}],
  "uncertain": [{"text": "строка", "segment_refs": ["S1"], "code_refs": ["C1"]}]
}"""

CONSOLIDATE_SYSTEM_PROMPT = """Ты — редактор протокола совещания. Тебе дан список тем, извлечённых из последовательных частей одного транскрипта.
Строгие правила:
1. Объедини близкие темы в крупные смысловые блоки. Не создавай больше тем, чем указано в запросе.
2. Не добавляй фактов, которых нет в исходных темах. Не придумывай имена, сроки и решения.
3. Для каждой темы объедини все ссылки на сегменты (segment_refs) и карточки экрана (code_refs) из исходных тем. Не теряй ссылки.
4. Верни только валидный JSON: {"topics": [{"title": "строка", "summary": "строка", "segment_refs": ["S1"], "code_refs": ["C1"]}]}"""

SUMMARY_SYSTEM_PROMPT = """Ты — редактор протокола совещания. Тебе дан список тем, решений, задач, рисков и вопросов, извлечённых из транскрипта.
Строгие правила:
1. Не добавляй фактов, которых нет в списке. Не придумывай имена, сроки и решения.
2. Сформулируй короткое название совещания и резюме.
3. Верни только валидный JSON без markdown-ограждений: {"title": "...", "summary": "..."}"""

ProgressFn = Callable[[str], None]


@dataclass(frozen=True)
class ReportLimits:
    max_topics: int
    max_items: int | None
    max_summary_sentences: int
    max_topic_summary_chars: int | None


def _limits(cfg: LLMConfig) -> ReportLimits:
    """Return the effective report profile.

    ``asd-ste100`` uses the short form configured in YAML. ``plain`` restores the
    full report used before the short-form change: 12 topics, no item cap, and
    a longer summary.
    """

    if cfg.report_style == "asd-ste100":
        return ReportLimits(
            max_topics=cfg.max_topics,
            max_items=cfg.max_items_per_section,
            max_summary_sentences=cfg.max_summary_sentences,
            max_topic_summary_chars=cfg.max_topic_summary_chars,
        )
    return ReportLimits(
        max_topics=12,
        max_items=None,
        max_summary_sentences=10,
        max_topic_summary_chars=None,
    )


STE_RULES = """Стиль изложения — ASD-STE100 (Simplified Technical English), адаптированный для русского языка:
- Пиши короткими предложениями: не более 20 слов.
- Одна мысль — одно предложение.
- Используй активный залог и простое настоящее время.
- Используй один и тот же термин для одного понятия; не используй синонимы.
- Не используй метафоры, рекламные оценки, канцелярит и вводные фразы.
- Не повторяй мысль разными словами.
- Заголовок — не более 10 слов.
- Каждый summary темы — не более 2 предложений.
- Резюме — не более 5 предложений.
- Каждый пункт содержит только факт и ссылки S<n>."""


def _with_style(system: str, cfg: LLMConfig) -> str:
    if cfg.report_style == "asd-ste100":
        return system + "\n\n" + STE_RULES
    return system


def _timestamp(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def _speaker_header(transcript: Transcript) -> str:
    labels = speaker_labels(transcript)
    parts = []
    for speaker in transcript.speakers:
        name = labels.get(speaker.speaker_id)
        parts.append(
            f"{speaker.speaker_id} ({name})" if name else speaker.speaker_id
        )
    return ", ".join(parts) if parts else "неизвестны"


def _segment_line(
    index: int, segment: Segment, labels: dict[str, str], use_names: bool = True
) -> str:
    speaker = speaker_label(segment.speaker_id, labels) if use_names else (segment.speaker_id or "unknown")
    if use_names and segment.speaker_id and segment.speaker_id in labels:
        speaker = f"{speaker} ({segment.speaker_id})"
    text = segment.corrected_text or segment.raw_text
    return (
        f"[S{index}] {speaker} "
        f"{_timestamp(segment.start)}-{_timestamp(segment.end)}: {text}"
    )


def transcript_prompt(
    transcript: Transcript,
    selected: list[tuple[int, Segment]] | None = None,
    max_chars: int = 120_000,
    use_names: bool = True,
    visual_context: str | None = None,
) -> str:
    labels = speaker_labels(transcript)
    lines = [
        f"Язык: {transcript.language}",
        f"Участники (speaker_id, не автоматические имена): {_speaker_header(transcript)}",
        "Транскрипт:",
    ]
    items = selected if selected is not None else list(enumerate(transcript.segments, start=1))
    used = 0
    for index, segment in items:
        line = _segment_line(index, segment, labels, use_names=use_names)
        if used + len(line) > max_chars:
            lines.append("[... фрагмент обрезан по лимиту ...]")
            break
        lines.append(line)
        used += len(line)
    if visual_context:
        lines += ["", "На экране (OCR, может содержать ошибки):", visual_context]
    return "\n".join(lines)


def chunk_segments(
    transcript: Transcript, max_chars: int, overlap_segments: int = 1
) -> list[list[tuple[int, Segment]]]:
    """Split the transcript into character-bounded chunks with a small overlap."""

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    labels = speaker_labels(transcript)
    chunks: list[list[tuple[int, Segment]]] = []
    current: list[tuple[int, Segment]] = []
    used = 0
    for index, segment in enumerate(transcript.segments, start=1):
        line_len = len(_segment_line(index, segment, labels, use_names=False)) + 1
        if current and used + line_len > max_chars:
            chunks.append(current)
            current = current[-overlap_segments:] if overlap_segments else []
            used = sum(
                len(_segment_line(i, s, labels, use_names=False)) + 1 for i, s in current
            )
        current.append((index, segment))
        used += line_len
    if current:
        chunks.append(current)
    return chunks


def _parse_json(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def _chat_json(
    client: LLMClient,
    system: str,
    user: str,
    retries: int,
) -> dict[str, Any] | None:
    for attempt in range(retries + 1):
        raw = client.chat(system, user, json_mode=True)
        payload = _parse_json(raw)
        if payload is not None:
            return payload
        if attempt < retries:
            system = system + "\nПредыдущий ответ не был валидным JSON. Верни только JSON."
    return None


def _as_float(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _sanitize_report(
    report: dict[str, Any],
    valid: set[str],
    valid_codes: set[str] | None = None,
) -> tuple[dict[str, Any], int]:
    dropped = 0
    valid_codes = valid_codes or set()

    def refs(value: Any, allowed: set[str]) -> list[str]:
        nonlocal dropped
        if not isinstance(value, list):
            return []
        kept = [str(v) for v in value if str(v) in allowed]
        dropped += len(value) - len(kept)
        return kept

    def text(value: Any) -> str:
        return str(value or "").strip()

    cleaned: dict[str, Any] = {
        "title": text(report.get("title")) or "Протокол совещания",
        "summary": text(report.get("summary")),
        "topics": [],
        "decisions": [],
        "action_items": [],
        "risks": [],
        "open_questions": [],
        "uncertain": [],
    }
    for item in report.get("topics") or []:
        if isinstance(item, dict):
            cleaned["topics"].append(
                {
                    "title": text(item.get("title")),
                    "summary": text(item.get("summary")),
                    "segment_refs": refs(item.get("segment_refs"), valid),
                    "code_refs": refs(item.get("code_refs"), valid_codes),
                }
            )
    for item in report.get("decisions") or []:
        if isinstance(item, dict):
            cleaned["decisions"].append(
                {
                    "text": text(item.get("text")),
                    "segment_refs": refs(item.get("segment_refs"), valid),
                    "code_refs": refs(item.get("code_refs"), valid_codes),
                    "confidence": _as_float(item.get("confidence")),
                }
            )
    for key in ("risks", "open_questions", "uncertain"):
        for item in report.get(key) or []:
            if isinstance(item, dict):
                cleaned[key].append(
                    {
                        "text": text(item.get("text")),
                        "segment_refs": refs(item.get("segment_refs"), valid),
                        "code_refs": refs(item.get("code_refs"), valid_codes),
                    }
                )
    for item in report.get("action_items") or []:
        if isinstance(item, dict):
            cleaned["action_items"].append(
                {
                    "task": text(item.get("task")),
                    "owner": item.get("owner"),
                    "due": item.get("due"),
                    "segment_refs": refs(item.get("segment_refs"), valid),
                    "code_refs": refs(item.get("code_refs"), valid_codes),
                    "confidence": _as_float(item.get("confidence")),
                }
            )
    return cleaned, dropped


def _norm_text(text: Any) -> str:
    return re.sub(r"[^\w]+", " ", str(text or "").casefold()).strip()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _merge_refs(left: list[str], right: list[str]) -> list[str]:
    def order(value: str) -> int:
        try:
            return int(value[1:])
        except (TypeError, ValueError):
            return 10**9

    return sorted(set(left) | set(right), key=order)


def _dedupe_items(
    items: list[dict[str, Any]], key: str, threshold: float
) -> list[dict[str, Any]]:
    """Deterministic de-duplication that never drops distinct facts."""

    result: list[dict[str, Any]] = []
    normalized: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = _norm_text(item.get(key))
        if not text:
            continue
        duplicate_index = None
        for index, existing in enumerate(normalized):
            if _similarity(text, existing) >= threshold:
                duplicate_index = index
                break
        if duplicate_index is None:
            result.append(dict(item))
            normalized.append(text)
        else:
            result[duplicate_index]["segment_refs"] = _merge_refs(
                list(result[duplicate_index].get("segment_refs") or []),
                list(item.get("segment_refs") or []),
            )
            result[duplicate_index]["code_refs"] = _merge_refs(
                list(result[duplicate_index].get("code_refs") or []),
                list(item.get("code_refs") or []),
            )
    return result


def _aggregate_parts(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate chunk fragments and remove near-duplicates without an LLM.

    Letting an LLM merge 20 fragments loses most items, so the reduce phase is
    deterministic. The LLM is only asked for a title and summary afterwards.
    """

    aggregated: dict[str, Any] = {
        "title": "",
        "summary": "",
        "topics": [],
        "decisions": [],
        "action_items": [],
        "risks": [],
        "open_questions": [],
        "uncertain": [],
    }
    for part in parts:
        for key in ("topics", "decisions", "action_items", "risks", "open_questions", "uncertain"):
            aggregated[key].extend(part.get(key) or [])

    aggregated["topics"] = _dedupe_items(aggregated["topics"], "title", 0.85)
    aggregated["decisions"] = _dedupe_items(aggregated["decisions"], "text", 0.9)
    aggregated["action_items"] = _dedupe_items(aggregated["action_items"], "task", 0.9)
    for key in ("risks", "open_questions", "uncertain"):
        aggregated[key] = _dedupe_items(aggregated[key], "text", 0.9)
    return aggregated


def _group_topics(topics: list[dict[str, Any]], max_chars: int) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    used = 0
    for topic in topics:
        size = len(json.dumps(topic, ensure_ascii=False))
        if current and used + size > max_chars:
            groups.append(current)
            current = []
            used = 0
        current.append(topic)
        used += size
    if current:
        groups.append(current)
    return groups


def _consolidate_group(
    client: LLMClient,
    topics: list[dict[str, Any]],
    limit: int,
    cfg: LLMConfig,
    valid: set[str],
    valid_codes: set[str],
    previous_count: int | None = None,
) -> list[dict[str, Any]] | None:
    payload = [
        {
            "title": item.get("title"),
            "summary": item.get("summary"),
            "segment_refs": item.get("segment_refs"),
            "code_refs": item.get("code_refs"),
        }
        for item in topics
    ]
    instruction = f"Объедини темы в не более {limit} крупных тем."
    if previous_count is not None:
        instruction += (
            f" Предыдущий ответ содержал {previous_count} тем — это слишком много."
            f" Объедини их в {limit} или меньше."
        )
    user = instruction + "\n" + json.dumps(payload, ensure_ascii=False)
    data = _chat_json(
        client, _with_style(CONSOLIDATE_SYSTEM_PROMPT, cfg), user, cfg.retries
    )
    if data is None or not isinstance(data.get("topics"), list):
        return None
    sanitized, _ = _sanitize_report({"topics": data["topics"]}, valid, valid_codes)
    return sanitized.get("topics") or None


def _merge_excess(topics: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Deterministically cap topics, preserving all references."""

    if len(topics) <= limit:
        return topics
    keep = topics[: max(1, limit - 1)]
    rest = topics[max(1, limit - 1) :]
    if not rest:
        return keep
    refs: list[str] = []
    code_refs: list[str] = []
    for item in rest:
        refs.extend(item.get("segment_refs") or [])
        code_refs.extend(item.get("code_refs") or [])
    titles = [str(item.get("title") or "").strip() for item in rest]
    summary = "Также обсуждались: " + "; ".join(title for title in titles if title)
    keep.append(
        {
            "title": "Прочие темы",
            "summary": summary[:4000],
            "segment_refs": sorted(set(refs), key=lambda value: int(value[1:]) if value[1:].isdigit() else 10**9),
            "code_refs": _merge_refs([], code_refs),
        }
    )
    return keep


def _consolidate_topics(
    client: LLMClient,
    report: dict[str, Any],
    cfg: LLMConfig,
    valid: set[str],
    valid_codes: set[str],
) -> list[str]:
    """Merge granular topics into the profile's target without exceeding context.

    Topics are consolidated in context-sized groups; the loop halves the number
    of topics until the target is reached or no further progress is made. A
    deterministic fallback guarantees the cap and preserves references.
    """

    target = _limits(cfg).max_topics
    warnings: list[str] = []
    topics = report.get("topics") or []
    for _ in range(4):
        if len(topics) <= target:
            break
        groups = _group_topics(topics, cfg.merge_chars)
        if len(groups) <= 1 and len(groups[0]) > 1:
            half = max(1, len(groups[0]) // 2)
            groups = [groups[0][:half], groups[0][half:]]
        consolidated: list[dict[str, Any]] = []
        for group in groups:
            share = max(1, round(target * len(group) / len(topics)))
            limit = max(1, min(target, share))
            merged = _consolidate_group(client, group, limit, cfg, valid, valid_codes)
            if merged is not None and len(merged) > limit:
                retried = _consolidate_group(
                    client, group, limit, cfg, valid, valid_codes, previous_count=len(merged)
                )
                if retried is not None:
                    merged = retried
            if merged is not None and len(merged) > limit:
                merged = _merge_excess(merged, limit)
                warnings.append(
                    f"topic consolidation capped a group to {limit} theme(s)"
                )
            consolidated.extend(merged or group)
        if len(consolidated) >= len(topics):
            topics = _merge_excess(consolidated, target)
            warnings.append("topic consolidation reached the deterministic cap")
            break
        topics = consolidated
    if len(topics) > target:
        topics = _merge_excess(topics, target)
        warnings.append("topic consolidation reached the deterministic cap")
    report["topics"] = topics
    return warnings


def _synthesize_summary(
    client: LLMClient,
    report: dict[str, Any],
    cfg: LLMConfig,
    labels: dict[str, str],
) -> None:
    payload = {
        "participants": [
            {
                "speaker_id": item.get("speaker_id"),
                "name": labels.get(str(item.get("speaker_id"))),
            }
            for item in report.get("participants", [])
        ],
        "topics": [item.get("title") for item in report.get("topics", [])][:60],
        "decisions": [item.get("text") for item in report.get("decisions", [])][:60],
        "action_items": [item.get("task") for item in report.get("action_items", [])][:60],
        "risks": [item.get("text") for item in report.get("risks", [])][:60],
        "open_questions": [item.get("text") for item in report.get("open_questions", [])][:60],
    }
    user = (
        "Данные протокола:\n"
        + json.dumps(payload, ensure_ascii=False)
        + f"\nВерни заголовок (не более 10 слов) и резюме не более "
        f"{_limits(cfg).max_summary_sentences} предложений."
    )
    if len(user) > cfg.merge_chars:
        payload["topics"] = payload["topics"][:20]
        payload["decisions"] = payload["decisions"][:30]
        payload["action_items"] = payload["action_items"][:30]
        user = "Данные протокола:\n" + json.dumps(payload, ensure_ascii=False)
    data = _chat_json(
        client, _with_style(SUMMARY_SYSTEM_PROMPT, cfg), user, cfg.retries
    )
    if data is None:
        report["title"] = report.get("title") or "Протокол совещания"
        return
    title = str(data.get("title") or "").strip()
    summary = str(data.get("summary") or "").strip()
    if title:
        report["title"] = title
    if summary:
        report["summary"] = summary


def _fill_code_refs_by_time(
    report: dict[str, Any],
    transcript: Transcript,
    cards: list[CodeCard],
    padding: float,
) -> None:
    """Attach visual cards to items that the LLM did not reference explicitly.

    The fallback is deterministic: it uses only the time range of the item's
    segment references. It prevents useful screen material from being lost when
    the model omits ``code_refs``.
    """

    if not cards:
        return
    segments = {f"S{index}": segment for index, segment in enumerate(transcript.segments, start=1)}
    for key in ("decisions", "action_items", "risks", "open_questions", "uncertain"):
        for item in report.get(key) or []:
            if item.get("code_refs"):
                continue
            refs = [segments[ref] for ref in item.get("segment_refs") or [] if ref in segments]
            if not refs:
                continue
            start = min(segment.start for segment in refs)
            end = max(segment.end for segment in refs)
            matches = [
                card
                for card in cards
                if card.kind in {"code", "slide"}
                and card.start <= end + padding
                and card.end >= start - padding
            ]
            if matches:
                item["code_refs"] = [card.code_id for card in matches[:2]]


def _humanize_speaker_ids(report: dict[str, Any], transcript: Transcript) -> None:
    labels = speaker_labels(transcript)

    def fix(text: str) -> str:
        def replacement(match: re.Match[str]) -> str:
            return labels.get(match.group(0), "участник")

        return re.sub(r"[Ss]peaker_?\d+", replacement, text)

    def walk(value: Any) -> Any:
        if isinstance(value, str):
            return fix(value)
        if isinstance(value, list):
            return [walk(item) for item in value]
        if isinstance(value, dict):
            return {key: walk(item) for key, item in value.items()}
        return value

    for key in ("title", "summary", "topics", "decisions", "action_items", "risks", "open_questions", "uncertain"):
        if key in report:
            report[key] = walk(report[key])


def _limit_report(report: dict[str, Any], cfg: LLMConfig) -> None:
    """Apply the selected profile's hard limits."""

    limits = _limits(cfg)
    if limits.max_items is not None:
        max_items = limits.max_items
        for key in ("decisions", "action_items"):
            items = report.get(key) or []
            items.sort(
                key=lambda item: float(item.get("confidence") or 0.0), reverse=True
            )
            report[key] = items[:max_items]
        for key in ("risks", "open_questions", "uncertain"):
            report[key] = (report.get(key) or [])[:max_items]

    if limits.max_topic_summary_chars is not None:
        for topic in report.get("topics") or []:
            topic["summary"] = _shorten_chars(
                str(topic.get("summary") or ""), limits.max_topic_summary_chars
            )
    report["summary"] = _shorten_sentences(
        str(report.get("summary") or ""), limits.max_summary_sentences
    )
    if limits.max_topic_summary_chars is not None:
        report["title"] = _shorten_chars(str(report.get("title") or ""), 140)


def _shorten_sentences(text: str, max_sentences: int) -> str:
    text = " ".join(text.split())
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentences[:max_sentences]).strip()


def _shorten_chars(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0].rstrip(".,;:")
    return cut + "…"


def _add_participants(report: dict[str, Any], transcript: Transcript) -> None:
    labels = speaker_labels(transcript)
    stats = collect_speaker_stats(transcript)
    report["participants"] = [
        {
            "speaker_id": entry.speaker_id,
            "name": labels.get(entry.speaker_id),
            "talk_time_seconds": round(entry.talk_time_seconds, 1),
            "segments": entry.segments,
        }
        for entry in stats
    ]


def generate_meeting_report(
    transcript: Transcript,
    cfg: LLMConfig,
    client: LLMClient | None = None,
    work_dir: str | Path | None = None,
    progress: ProgressFn | None = None,
    code_cards: list[CodeCard] | None = None,
    visual_context_chars: int = 1200,
    visual_align_window: float = 30.0,
) -> tuple[dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    if not cfg.enabled:
        return None, warnings
    if not transcript.segments:
        return None, ["meeting report skipped: transcript has no segments"]

    code_cards = code_cards or []
    client = client or LLMClient(cfg)
    chunks = chunk_segments(transcript, cfg.chunk_chars)
    if not chunks:
        return None, ["meeting report skipped: nothing to summarise"]

    valid = {f"S{i}" for i in range(1, len(transcript.segments) + 1)}
    valid_codes = {card.code_id for card in code_cards}
    visual_fingerprint = hashlib.sha256(
        json.dumps(
            [card.to_dict() for card in code_cards],
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:12]
    if work_dir:
        key = hashlib.sha256(
            json.dumps(
                {
                    "model": cfg.model,
                    "chunk_chars": cfg.chunk_chars,
                    "prompt": "ids-v2-visual",
                    "visual": visual_fingerprint,
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()[:12]
        chunk_dir = Path(work_dir) / "report_chunks" / key
        chunk_dir.mkdir(parents=True, exist_ok=True)
    else:
        chunk_dir = None

    parts: list[dict[str, Any]] = []
    try:
        for number, chunk in enumerate(chunks, start=1):
            if progress:
                progress(f"LLM report: chunk {number}/{len(chunks)}")
            cache_path = chunk_dir / f"chunk_{number:03d}.json" if chunk_dir else None
            payload: dict[str, Any] | None = None
            if cache_path and cache_path.is_file():
                payload = _parse_json(cache_path.read_text(encoding="utf-8"))
            if payload is None:
                chunk_start = min(segment.start for _, segment in chunk)
                chunk_end = max(segment.end for _, segment in chunk)
                visual_context = build_visual_context(
                    code_cards,
                    chunk_start,
                    chunk_end,
                    visual_context_chars,
                    padding=visual_align_window,
                )
                payload = _chat_json(
                    client,
                    _with_style(SYSTEM_PROMPT, cfg),
                    transcript_prompt(
                        transcript,
                        selected=chunk,
                        max_chars=cfg.chunk_chars,
                        use_names=False,
                        visual_context=visual_context,
                    ),
                    cfg.retries,
                )
                if payload is not None and cache_path:
                    cache_path.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
            if payload is None:
                warnings.append(
                    f"meeting report: chunk {number} skipped (invalid JSON after retries)"
                )
                continue
            sanitized, dropped = _sanitize_report(payload, valid, valid_codes)
            if dropped:
                warnings.append(
                    f"meeting report: chunk {number} dropped {dropped} unknown reference(s)"
                )
            parts.append(sanitized)

        if not parts:
            return None, warnings + ["meeting report skipped: all chunks failed"]

        if progress:
            progress(f"LLM report: aggregating {len(parts)} part(s)")
        merged = _aggregate_parts(parts)
        if progress:
            progress("LLM report: consolidating topics")
        warnings.extend(_consolidate_topics(client, merged, cfg, valid, valid_codes))
        _add_participants(merged, transcript)
        merged["code_cards"] = [card.to_dict() for card in code_cards]
        if progress:
            progress("LLM report: title and summary")
        _synthesize_summary(client, merged, cfg, speaker_labels(transcript))
        _humanize_speaker_ids(merged, transcript)
        _limit_report(merged, cfg)
        _fill_code_refs_by_time(
            merged, transcript, code_cards, visual_align_window
        )

        if not merged.get("title") or merged["title"] == "Протокол совещания":
            merged["title"] = f"Протокол совещания: {transcript.meeting_id}"
        return merged, warnings
    except PipelineError as exc:
        return None, warnings + [f"meeting report skipped: {exc}"]
    finally:
        client.unload()


def _format_refs(refs: list[str], max_ranges: int = 8) -> str:
    """Compact ``S2-S22, S47-S53`` rendering; the JSON keeps every id."""

    numbers = sorted(
        {int(value[1:]) for value in refs if re.fullmatch(r"S\d+", value)}
    )
    if not numbers:
        return ""
    ranges: list[str] = []
    start = previous = numbers[0]
    for number in numbers[1:]:
        if number == previous + 1:
            previous = number
            continue
        ranges.append(f"S{start}" if start == previous else f"S{start}-S{previous}")
        start = previous = number
    ranges.append(f"S{start}" if start == previous else f"S{start}-S{previous}")
    if len(ranges) > max_ranges:
        ranges = ranges[:max_ranges] + ["…"]
    return ", ".join(ranges)


def _format_code_refs(refs: list[str], cards: dict[str, dict[str, Any]], max_items: int = 8) -> str:
    parts: list[str] = []
    for ref in refs:
        if not re.fullmatch(r"C\d+", ref):
            continue
        card = cards.get(ref)
        if card and card.get("start") is not None:
            parts.append(f"{ref} ({_timestamp(float(card['start']))})")
        else:
            parts.append(ref)
    if len(parts) > max_items:
        parts = parts[:max_items] + ["…"]
    return ", ".join(parts)


def _card_index(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(card.get("code_id")): card
        for card in report.get("code_cards") or []
        if isinstance(card, dict) and card.get("code_id")
    }


def _source_suffix(item: dict[str, Any], cards: dict[str, dict[str, Any]]) -> str:
    parts: list[str] = []
    segment_refs = item.get("segment_refs") or []
    if segment_refs:
        parts.append(_format_refs(segment_refs))
    code_refs = item.get("code_refs") or []
    if code_refs:
        rendered = _format_code_refs(code_refs, cards)
        if rendered:
            parts.append(f"код: {rendered}")
    return "; ".join(part for part in parts if part)


def render_report_markdown(
    report: dict[str, Any], frames_dir: str | None = None
) -> str:
    cards = _card_index(report)
    lines = [f"# {report.get('title', 'Протокол совещания')}", ""]
    participants = report.get("participants") or []
    if participants:
        lines += ["## Участники", ""]
        for item in participants:
            name = item.get("name") or item.get("speaker_id")
            minutes = round(float(item.get("talk_time_seconds") or 0) / 60, 1)
            lines.append(
                f"- **{name}** (`{item.get('speaker_id')}`) — {minutes} мин, "
                f"{item.get('segments', 0)} реплик"
            )
        lines.append("")
    if report.get("summary"):
        lines += ["## Краткое резюме", "", report["summary"], ""]
    if report.get("topics"):
        lines += ["## Обсуждение по темам", ""]
        for topic in report["topics"]:
            lines.append(f"### {topic.get('title') or 'Без названия'}")
            lines.append(topic.get("summary") or "")
            sources = _source_suffix(topic, cards)
            if sources:
                lines.append(f"Источники: {sources}")
            lines.append("")
    for key, title in (
        ("decisions", "Решения"),
        ("action_items", "Задачи"),
        ("risks", "Риски"),
        ("open_questions", "Открытые вопросы"),
        ("uncertain", "Требует проверки"),
    ):
        items = report.get(key) or []
        if not items:
            continue
        lines += [f"## {title}", ""]
        for item in items:
            if key == "action_items":
                text = item.get("task") or ""
                extras = []
                if item.get("owner"):
                    extras.append(f"ответственный: {item['owner']}")
                if item.get("due"):
                    extras.append(f"срок: {item['due']}")
                if extras:
                    text += f" ({'; '.join(extras)})"
            else:
                text = item.get("text") or ""
            sources = _source_suffix(item, cards)
            suffix = f" — {sources}" if sources else ""
            confidence = item.get("confidence")
            if confidence is not None and confidence < 0.6:
                suffix += " (низкая уверенность)"
            lines.append(f"- {text}{suffix}")
        lines.append("")

    shown = [
        card
        for card in report.get("code_cards") or []
        if isinstance(card, dict) and str(card.get("text") or "").strip()
    ]
    if shown:
        lines += ["## Код и материалы на экране", ""]
        lines.append(
            "Текст распознан OCR с кадров демонстрации экрана и может содержать ошибки. "
            "Ссылки на карточки C<n> в разделах выше указывают на эти материалы."
        )
        lines.append("")
        for card in shown:
            code_id = str(card.get("code_id"))
            kind = "код" if card.get("kind") == "code" else "материал"
            start = float(card.get("start") or 0.0)
            end = float(card.get("end") or start)
            lines.append(f"### {code_id} — {_timestamp(start)}–{_timestamp(end)} ({kind})")
            frame = card.get("frame")
            if frames_dir and frame:
                lines += [f"![Кадр {code_id}]({frames_dir}/{frame})", ""]
            text = str(card.get("text") or "").rstrip()
            fence = "````" if "```" in text else "```"
            lines += [f"{fence}text", text, fence, ""]
    return "\n".join(lines).rstrip() + "\n"
