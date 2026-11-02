import json

from meeting_transcriber.output.json_writer import transcript_to_json, write_json
from meeting_transcriber.output.markdown_writer import render_markdown
from meeting_transcriber.output.srt_writer import render_srt
from meeting_transcriber.types import AudioMetadata, Correction, Segment, Speaker, Transcript


def _transcript() -> Transcript:
    return Transcript(
        meeting_id="meeting_001",
        language="ru",
        source_audio=AudioMetadata(
            filename="meeting.wav", duration=16.82, sample_rate=48000, channels=2
        ),
        speakers=[Speaker("speaker_00")],
        segments=[
            Segment(
                start=12.34,
                end=16.82,
                speaker_id="speaker_00",
                raw_text="Нужно проверить витэка чарт икс уай.",
                corrected_text="Нужно проверить vtkChartXY.",
                corrections=[
                    Correction(
                        source="витэка чарт икс уай",
                        target="vtkChartXY",
                        term_id="vtkChartXY",
                        confidence=0.93,
                    )
                ],
            )
        ],
        models={"asr": "whisper", "diarization": "titanet"},
    )


def test_json_contract():
    data = json.loads(transcript_to_json(_transcript()))
    assert data["schema_version"] == "1.0"
    assert data["segments"][0]["corrected_text"] == "Нужно проверить vtkChartXY."
    assert data["segments"][0]["corrections"][0]["confidence"] == 0.93
    assert data["source_audio"]["duration"] == 16.82


def test_json_can_drop_words():
    data = json.loads(transcript_to_json(_transcript(), include_words=False))
    assert "words" not in data["segments"][0]


def test_markdown_and_srt(tmp_path):
    transcript = _transcript()
    md = render_markdown(transcript)
    assert "vtkChartXY" in md
    assert "00:00:12 speaker_00" in md
    srt = render_srt(transcript)
    assert "00:00:12,340 --> 00:00:16,820" in srt
    assert "[speaker_00]" in srt


def test_writers_are_atomic(tmp_path):
    path = write_json(_transcript(), tmp_path / "out.json")
    assert path.is_file()
    assert not list(tmp_path.glob("*.tmp"))
