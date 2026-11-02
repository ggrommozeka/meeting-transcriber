from meeting_transcriber.output.markdown_writer import render_markdown
from meeting_transcriber.output.srt_writer import render_srt
from meeting_transcriber.speakers import (
    apply_speaker_names,
    collect_speaker_stats,
    load_speaker_map,
    speaker_labels,
    write_speaker_template,
)
from meeting_transcriber.types import AudioMetadata, Segment, Speaker, Transcript


def _transcript() -> Transcript:
    return Transcript(
        meeting_id="m1",
        language="ru",
        source_audio=AudioMetadata(filename="m1.wav", duration=20.0),
        speakers=[Speaker("speaker_00"), Speaker("speaker_01")],
        segments=[
            Segment(0.0, 6.0, "speaker_00", "Добрый день, начнём совещание."),
            Segment(6.0, 12.0, "speaker_01", "Я подготовил отчёт по проекту."),
            Segment(12.0, 13.0, "speaker_01", "Да."),
        ],
    )


def test_speaker_stats_and_template(tmp_path):
    transcript = _transcript()
    stats = collect_speaker_stats(transcript)
    by_id = {entry.speaker_id: entry for entry in stats}
    assert by_id["speaker_00"].talk_time_seconds == 6.0
    assert by_id["speaker_01"].segments == 2
    # Short "Да." is not used as a sample.
    assert all(sample != "Да." for sample in by_id["speaker_01"].samples)

    template = tmp_path / "m1.speakers.yaml"
    path = write_speaker_template(transcript, template)
    text = path.read_text(encoding="utf-8")
    assert "speaker_00" in text
    assert "name: ''" in text or 'name: ""' in text


def test_speaker_map_accepts_template_and_flat_formats(tmp_path):
    flat = tmp_path / "flat.yaml"
    flat.write_text(
        "speakers:\n  speaker_00: Иван\n  speaker_01: Мария\n", encoding="utf-8"
    )
    assert load_speaker_map(flat) == {"speaker_00": "Иван", "speaker_01": "Мария"}

    nested = tmp_path / "nested.yaml"
    nested.write_text(
        "meeting_id: m1\nspeakers:\n"
        "  speaker_00:\n    name: Иван\n    samples: []\n"
        "  speaker_01:\n    name: ''\n",
        encoding="utf-8",
    )
    assert load_speaker_map(nested) == {"speaker_00": "Иван"}


def test_display_names_in_outputs(tmp_path):
    transcript = _transcript()
    applied = apply_speaker_names(transcript, {"speaker_00": "Иван", "speaker_01": "Мария"})
    assert applied == 2
    assert speaker_labels(transcript) == {"speaker_00": "Иван", "speaker_01": "Мария"}

    markdown = render_markdown(transcript)
    assert "Иван (speaker_00)" in markdown
    assert "Мария (speaker_01)" in markdown
    srt = render_srt(transcript)
    assert "[Иван]" in srt
    assert "[Мария]" in srt
    # Speaker ids remain in JSON for auditability.
    data = transcript.to_dict()
    assert data["speakers"][0]["display_name"] == "Иван"
    assert data["segments"][0]["speaker_id"] == "speaker_00"


def test_missing_map_is_rejected(tmp_path):
    import pytest

    from meeting_transcriber.errors import ConfigError

    with pytest.raises(ConfigError):
        load_speaker_map(tmp_path / "missing.yaml")
