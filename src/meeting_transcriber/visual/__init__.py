"""Optional video frame analysis.

The visual stage extracts stable screen states (slides, code, diagrams) from a
video, OCRs them locally and exposes time-aligned ``CodeCard`` records. The
report pipeline uses the cards as auxiliary context and links report items back
to the code shown on screen.
"""

from .context import build_visual_context, cards_for_range
from .extract import VisualResult, extract_visual, load_visual_result
from .types import CodeCard

__all__ = [
    "CodeCard",
    "VisualResult",
    "build_visual_context",
    "cards_for_range",
    "extract_visual",
    "load_visual_result",
]
