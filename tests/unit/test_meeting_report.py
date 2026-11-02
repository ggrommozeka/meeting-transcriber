import json

from meeting_transcriber.config import LLMConfig
from meeting_transcriber.postprocessing.meeting_report import (
    chunk_segments,
    generate_meeting_report,
    render_report_markdown,
    transcript_prompt,
)
from meeting_transcriber.speakers import apply_speaker_names
from meeting_transcriber.types import AudioMetadata, Segment, Speaker, Transcript


class FakeLLM:
    def __init__(self, merge_refs=None):
        self.calls = []
        self.unloaded = False
        self.merge_refs = merge_refs or ["S1", "S2"]
        self.chunk_index = 0

    def chat(self, system, user, json_mode=False):
        self.calls.append((system, user))
        if "Объедини близкие темы" in system:
            return json.dumps(
                {
                    "topics": [
                        {
                            "title": "Сводная тема",
                            "summary": "Объединённое обсуждение.",
                            "segment_refs": self.merge_refs,
                        }
                    ]
                }
            )
        if "редактор протокола" in system:
            return json.dumps(
                {
                    "title": "Совещание по проекту",
                    "summary": "Обсудили статус проекта.",
                    "topics": [
                        {
                            "title": "Статус",
                            "summary": "Отчёт готов.",
                            "segment_refs": self.merge_refs,
                        }
                    ],
                    "decisions": [
                        {
                            "text": "Продолжаем работу.",
                            "segment_refs": ["S1", "S999"],
                            "confidence": 0.9,
                        }
                    ],
                    "action_items": [
                        {
                            "task": "Подготовить отчёт",
                            "owner": "Иван",
                            "due": None,
                            "segment_refs": ["S2"],
                            "confidence": 0.8,
                        }
                    ],
                    "risks": [],
                    "open_questions": [],
                    "uncertain": [],
                }
            )
        self.chunk_index += 1
        return json.dumps(
            {
                "topics": [
                    {
                        "title": f"Тема {self.chunk_index}",
                        "summary": "Кратко.",
                        "segment_refs": ["S1", "S999"],
                    }
                ],
                "decisions": [
                    {
                        "text": "Продолжаем работу.",
                        "segment_refs": ["S1", "S999"],
                        "confidence": 0.9,
                    }
                ],
                "action_items": [
                    {
                        "task": "Подготовить отчёт",
                        "owner": "Иван",
                        "due": None,
                        "segment_refs": ["S2"],
                        "confidence": 0.8,
                    }
                ],
                "risks": [],
                "open_questions": [],
                "uncertain": [],
            }
        )

    def unload(self):
        self.unloaded = True


def _transcript() -> Transcript:
    segments = [
        Segment(
            float(i * 5),
            float(i * 5 + 4),
            "speaker_00" if i % 2 == 0 else "speaker_01",
            f"Реплика номер {i} про проект и сроки, достаточно длинная для чанка.",
        )
        for i in range(10)
    ]
    return Transcript(
        meeting_id="m1",
        language="ru",
        source_audio=AudioMetadata(filename="m1.wav", duration=50.0),
        speakers=[Speaker("speaker_00"), Speaker("speaker_01")],
        segments=segments,
    )


def test_chunking_splits_and_overlaps():
    transcript = _transcript()
    chunks = chunk_segments(transcript, max_chars=400, overlap_segments=1)
    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert previous[-1][0] == current[0][0]  # one shared segment


def test_prompt_uses_display_names():
    transcript = _transcript()
    apply_speaker_names(transcript, {"speaker_00": "Иван"})
    prompt = transcript_prompt(transcript, selected=[(1, transcript.segments[0])])
    assert "Иван (speaker_00)" in prompt
    assert "[S1]" in prompt


def test_map_reduce_report_and_reference_validation():
    transcript = _transcript()
    apply_speaker_names(transcript, {"speaker_00": "Иван", "speaker_01": "Мария"})
    client = FakeLLM()
    cfg = LLMConfig(
        enabled=True,
        chunk_chars=400,
        merge_chars=100_000,
        retries=0,
    )
    report, warnings = generate_meeting_report(transcript, cfg, client=client)
    assert report is not None
    assert client.unloaded is True
    assert len(client.calls) > 2
    assert report["title"] == "Совещание по проекту"
    assert report["participants"][0]["name"] in {"Иван", "Мария"}
    # Unknown S999 references were dropped.
    assert report["decisions"][0]["segment_refs"] == ["S1"]
    assert any("unknown reference" in warning for warning in warnings)


def test_report_markdown_sections():
    transcript = _transcript()
    client = FakeLLM()
    cfg = LLMConfig(enabled=True, chunk_chars=400, merge_chars=100_000, retries=0)
    report, _ = generate_meeting_report(transcript, cfg, client=client)
    markdown = render_report_markdown(report)
    assert "## Участники" in markdown
    assert "## Краткое резюме" in markdown
    assert "## Решения" in markdown
    assert "## Задачи" in markdown
    assert "S1" in markdown


def test_topics_are_consolidated_when_too_many():
    transcript = _transcript()
    client = FakeLLM()
    cfg = LLMConfig(
        enabled=True,
        chunk_chars=400,
        merge_chars=100_000,
        max_topics=2,
        retries=0,
    )
    report, _ = generate_meeting_report(transcript, cfg, client=client)
    assert report is not None
    assert len(report["topics"]) <= 2


def test_compact_reference_rendering():
    from meeting_transcriber.postprocessing.meeting_report import _format_refs

    assert _format_refs(["S1", "S2", "S3", "S10"]) == "S1-S3, S10"
    assert _format_refs([]) == ""


def test_report_disabled_returns_nothing():
    report, warnings = generate_meeting_report(
        _transcript(), LLMConfig(enabled=False), client=FakeLLM()
    )
    assert report is None
    assert warnings == []
