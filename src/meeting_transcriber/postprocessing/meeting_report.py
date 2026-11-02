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
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ..config import LLMConfig
from ..errors import PipelineError
from ..speakers import collect_speaker_stats, speaker_label, speaker_labels
from ..types import Segment, Transcript
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
8. Верни только валидный JSON без markdown-ограждений.

Схема ответа:
{
  "topics": [{"title": "строка", "summary": "строка", "segment_refs": ["S1"]}],
  "decisions": [{"text": "строка", "segment_refs": ["S1"], "confidence": 0.0}],
  "action_items": [{"task": "строка", "owner": null, "due": null, "segment_refs": ["S1"], "confidence": 0.0}],
  "risks": [{"text": "строка", "segment_refs": ["S1"]}],
  "open_questions": [{"text": "строка", "segment_refs": ["S1"]}],
  "uncertain": [{"text": "строка", "segment_refs": ["S1"]}]
}"""

CONSOLIDATE_SYSTEM_PROMPT = """Ты — редактор протокола совещания. Тебе дан список тем, извлечённых из последовательных частей одного транскрипта.
Строгие правила:
1. Объедини близкие темы в крупные смысловые блоки. Не создавай больше тем, чем указано в запросе.
2. Не добавляй фактов, которых нет в исходных темах. Не придумывай имена, сроки и решения.
3. Для каждой темы объедини все ссылки на сегменты (segment_refs) из исходных тем. Не теряй ссылки.
4. Верни только валидный JSON: {"topics": [{"title": "строка", "summary": "строка", "segment_refs": ["S1"]}]}"""

SUMMARY_SYSTEM_PROMPT = """Ты — редактор протокола совещания. Тебе дан список тем, решений, задач, рисков и вопросов, извлечённых из транскрипта.
Строгие правила:
1. Не добавляй фактов, которых нет в списке. Не придумывай имена, сроки и решения.
2. Сформулируй короткое название совещания и резюме на 5-10 предложений.
3. Верни только валидный JSON без markdown-ограждений: {"title": "...", "summary": "..."}"""

ProgressFn = Callable[[str], None]


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
    report: dict[str, Any], valid: set[str]
) -> tuple[dict[str, Any], int]:
    dropped = 0

    def refs(value: Any) -> list[str]:
        nonlocal dropped
        if not isinstance(value, list):
            return []
        kept = [str(v) for v in value if str(v) in valid]
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
                    "segment_refs": refs(item.get("segment_refs")),
                }
            )
    for item in report.get("decisions") or []:
        if isinstance(item, dict):
            cleaned["decisions"].append(
                {
                    "text": text(item.get("text")),
                    "segment_refs": refs(item.get("segment_refs")),
                    "confidence": _as_float(item.get("confidence")),
                }
            )
    for key in ("risks", "open_questions", "uncertain"):
        for item in report.get(key) or []:
            if isinstance(item, dict):
                cleaned[key].append(
                    {
                        "text": text(item.get("text")),
                        "segment_refs": refs(item.get("segment_refs")),
                    }
                )
    for item in report.get("action_items") or []:
        if isinstance(item, dict):
            cleaned["action_items"].append(
                {
                    "task": text(item.get("task")),
                    "owner": item.get("owner"),
                    "due": item.get("due"),
                    "segment_refs": refs(item.get("segment_refs")),
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
) -> list[dict[str, Any]] | None:
    payload = [
        {
            "title": item.get("title"),
            "summary": item.get("summary"),
            "segment_refs": item.get("segment_refs"),
        }
        for item in topics
    ]
    user = (
        f"Объедини темы в не более {limit} крупных тем.\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    data = _chat_json(client, CONSOLIDATE_SYSTEM_PROMPT, user, cfg.retries)
    if data is None or not isinstance(data.get("topics"), list):
        return None
    sanitized, _ = _sanitize_report({"topics": data["topics"]}, valid)
    return sanitized.get("topics") or None


def _consolidate_topics(
    client: LLMClient,
    report: dict[str, Any],
    cfg: LLMConfig,
    valid: set[str],
) -> None:
    """Merge granular topics into ``max_topics`` without exceeding context.

    Topics are consolidated in context-sized groups; the loop halves the number
    of topics until the target is reached or no further progress is made.
    """

    topics = report.get("topics") or []
    for _ in range(4):
        if len(topics) <= cfg.max_topics:
            break
        groups = _group_topics(topics, cfg.merge_chars)
        if len(groups) <= 1 and len(groups[0]) > 1:
            # One oversized group: split in half by count.
            half = max(1, len(groups[0]) // 2)
            groups = [groups[0][:half], groups[0][half:]]
        consolidated: list[dict[str, Any]] = []
        for group in groups:
            share = max(1, round(cfg.max_topics * len(group) / len(topics)))
            limit = max(1, min(cfg.max_topics, share))
            merged = _consolidate_group(client, group, limit, cfg, valid)
            consolidated.extend(merged or group)
        if len(consolidated) >= len(topics):
            break
        topics = consolidated
    report["topics"] = topics


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
    user = "Данные протокола:\n" + json.dumps(payload, ensure_ascii=False)
    if len(user) > cfg.merge_chars:
        payload["topics"] = payload["topics"][:20]
        payload["decisions"] = payload["decisions"][:30]
        payload["action_items"] = payload["action_items"][:30]
        user = "Данные протокола:\n" + json.dumps(payload, ensure_ascii=False)
    data = _chat_json(client, SUMMARY_SYSTEM_PROMPT, user, cfg.retries)
    if data is None:
        report["title"] = report.get("title") or "Протокол совещания"
        return
    title = str(data.get("title") or "").strip()
    summary = str(data.get("summary") or "").strip()
    if title:
        report["title"] = title
    if summary:
        report["summary"] = summary


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
) -> tuple[dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    if not cfg.enabled:
        return None, warnings
    if not transcript.segments:
        return None, ["meeting report skipped: transcript has no segments"]

    client = client or LLMClient(cfg)
    chunks = chunk_segments(transcript, cfg.chunk_chars)
    if not chunks:
        return None, ["meeting report skipped: nothing to summarise"]

    valid = {f"S{i}" for i in range(1, len(transcript.segments) + 1)}
    if work_dir:
        key = hashlib.sha256(
            json.dumps(
                {
                    "model": cfg.model,
                    "chunk_chars": cfg.chunk_chars,
                    "prompt": "ids-v1",
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
                payload = _chat_json(
                    client,
                    SYSTEM_PROMPT,
                    transcript_prompt(
                        transcript,
                        selected=chunk,
                        max_chars=cfg.chunk_chars,
                        use_names=False,
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
            sanitized, dropped = _sanitize_report(payload, valid)
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
        _consolidate_topics(client, merged, cfg, valid)
        _add_participants(merged, transcript)
        if progress:
            progress("LLM report: title and summary")
        _synthesize_summary(client, merged, cfg, speaker_labels(transcript))
        _humanize_speaker_ids(merged, transcript)

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


def render_report_markdown(report: dict[str, Any]) -> str:
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
            if topic.get("segment_refs"):
                lines.append(f"Источники: {_format_refs(topic['segment_refs'])}")
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
            suffix = (
                f" — {_format_refs(item.get('segment_refs') or [])}"
                if item.get("segment_refs")
                else ""
            )
            confidence = item.get("confidence")
            if confidence is not None and confidence < 0.6:
                suffix += " (низкая уверенность)"
            lines.append(f"- {text}{suffix}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
