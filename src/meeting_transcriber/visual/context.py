"""Time-aligned visual context for LLM report prompts."""

from __future__ import annotations

from .types import CodeCard


def _timestamp(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def cards_for_range(
    cards: list[CodeCard],
    start: float,
    end: float,
    padding: float = 0.0,
) -> list[CodeCard]:
    """Return cards whose interval overlaps ``[start, end]`` plus padding."""

    left = start - padding
    right = end + padding
    return [
        card
        for card in cards
        if card.kind in {"code", "slide"} and card.start <= right and card.end >= left
    ]


def build_visual_context(
    cards: list[CodeCard],
    start: float,
    end: float,
    max_chars: int,
    padding: float = 30.0,
) -> str:
    """Render a compact, token-bounded context block for one transcript chunk."""

    if max_chars <= 0:
        return ""
    selected = cards_for_range(cards, start, end, padding)
    if not selected:
        return ""
    # Code is more useful than slide prose; keep the block budget for it first.
    selected.sort(key=lambda card: (card.kind != "code", card.start))

    blocks: list[str] = []
    used = 0
    for card in selected:
        label = "код" if card.kind == "code" else "материал"
        header = (
            f"[{card.code_id}] {_timestamp(card.start)}-{_timestamp(card.end)} "
            f"({label}, OCR):"
        )
        body = card.text.strip()
        if not body:
            continue
        remaining = max_chars - used - len(header) - 2
        if remaining <= 0:
            break
        if len(body) > remaining:
            body = body[: max(0, remaining - 1)].rstrip() + "…"
        block = f"{header}\n{body}"
        blocks.append(block)
        used += len(block) + 2
    return "\n\n".join(blocks)


__all__ = ["build_visual_context", "cards_for_range"]
