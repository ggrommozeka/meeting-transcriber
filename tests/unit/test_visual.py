import json

from PIL import Image, ImageDraw

from meeting_transcriber.config import LLMConfig
from meeting_transcriber.postprocessing.meeting_report import (
    generate_meeting_report,
    render_report_markdown,
)
from meeting_transcriber.types import AudioMetadata, Segment, Transcript
from meeting_transcriber.visual import CodeCard, build_visual_context
from meeting_transcriber.visual.context import cards_for_range
from meeting_transcriber.visual.extract import classify_text
from meeting_transcriber.visual.frames import FrameSample, dhash, group_samples


def _image(path, shade):
    image = Image.new("RGB", (120, 80), (shade, shade, shade))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 90, 40), fill=(255 - shade, 0, 0))
    image.save(path)
    return path


def test_dhash_and_grouping(tmp_path):
    paths = []
    for index, shade in enumerate((10, 12, 200)):
        path = tmp_path / f"f{index}.jpg"
        _image(path, shade)
        paths.append(path)
    assert dhash(paths[0]) != dhash(paths[2])

    samples = [
        FrameSample(timestamp=0.0, path=paths[0], sharpness=1.0),
        FrameSample(timestamp=2.0, path=paths[1], sharpness=3.0),
        FrameSample(timestamp=4.0, path=paths[2], sharpness=2.0),
    ]
    groups = group_samples(samples, duration=6.0, dedupe_distance=6)
    assert len(groups) == 2
    assert groups[0].sample.path in {paths[0], paths[1]}
    assert groups[0].start == 0.0
    assert groups[0].end == 4.0
    assert groups[1].end == 6.0


def test_classify_text():
    code = (
        "class Solver {\n"
        "public:\n"
        "  virtual ~Solver() = default;\n"
        "  void Step(double dt);\n"
        "};"
    )
    assert classify_text(code, min_chars=20) == "code"
    assert (
        classify_text("Обсуждение архитектуры решателя и планов на неделю", min_chars=10)
        == "slide"
    )
    assert classify_text("ok", min_chars=20) == "noise"


def test_visual_context_is_time_bounded():
    cards = [
        CodeCard("C1", 10.0, 20.0, "code", "class A {};"),
        CodeCard("C2", 100.0, 110.0, "slide", "План работ"),
    ]
    selected = cards_for_range(cards, 0.0, 30.0)
    assert [card.code_id for card in selected] == ["C1"]
    context = build_visual_context(cards, 0.0, 30.0, max_chars=500, padding=0)
    assert "[C1]" in context
    assert "class A {};" in context
    assert "C2" not in context


class FakeLLM:
    def chat(self, system, user, json_mode=False):
        if "Объедини близкие темы" in system or "редактор протокола" in system:
            return json.dumps(
                {
                    "title": "Решатель",
                    "summary": "Обсудили код.",
                    "topics": [
                        {
                            "title": "Код",
                            "summary": "Показали класс.",
                            "segment_refs": ["S1"],
                            "code_refs": ["C1", "C99"],
                        }
                    ],
                    "decisions": [
                        {
                            "text": "Использовать класс Solver.",
                            "segment_refs": ["S1"],
                            "confidence": 0.9,
                        }
                    ],
                    "action_items": [],
                    "risks": [],
                    "open_questions": [],
                    "uncertain": [],
                }
            )
        return json.dumps(
            {
                "topics": [
                    {
                        "title": "Код",
                        "summary": "Показали класс.",
                        "segment_refs": ["S1"],
                        "code_refs": ["C1", "C99"],
                    }
                ],
                "decisions": [
                    {
                        "text": "Использовать класс Solver.",
                        "segment_refs": ["S1"],
                        "confidence": 0.9,
                    }
                ],
                "action_items": [],
                "risks": [],
                "open_questions": [],
                "uncertain": [],
            }
        )

    def unload(self):
        pass


def _transcript():
    return Transcript(
        meeting_id="m1",
        language="ru",
        source_audio=AudioMetadata(filename="m1.mp4", duration=10.0),
        segments=[Segment(0.0, 5.0, "speaker_00", "Здесь показан класс Solver.")],
    )


def test_report_keeps_known_code_refs_and_renders_section():
    card = CodeCard("C1", 0.0, 5.0, "code", "class Solver {};", frame="C1.jpg")
    cfg = LLMConfig(enabled=True, chunk_chars=10_000, merge_chars=100_000, retries=0)
    report, warnings = generate_meeting_report(
        _transcript(), cfg, client=FakeLLM(), code_cards=[card]
    )
    assert report is not None
    assert report["topics"][0]["code_refs"] == ["C1"]
    assert report["decisions"][0]["code_refs"] == ["C1"]
    assert any("unknown reference" in warning for warning in warnings)
    assert report["code_cards"][0]["code_id"] == "C1"

    markdown = render_report_markdown(report, frames_dir="m1.frames")
    assert "код: C1 (00:00:00)" in markdown
    assert "## Код и материалы на экране" in markdown
    assert "![Кадр C1](m1.frames/C1.jpg)" in markdown
    assert "class Solver {};" in markdown
