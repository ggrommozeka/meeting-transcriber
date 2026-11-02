from pathlib import Path

import pytest

from meeting_transcriber.asr.base import ASRResult
from meeting_transcriber.audio.preprocess import AudioChunk, PreparedAudio
from meeting_transcriber.config import load_config
from meeting_transcriber.diarization.base import DiarizationResult
from meeting_transcriber.pipeline import runner
from meeting_transcriber.types import AudioMetadata, DiarizationInterval, Segment

ROOT = Path(__file__).resolve().parents[2]


class FakeWhisper:
    calls = 0

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def transcribe(self, chunks, language="ru", initial_prompt=None):
        FakeWhisper.calls += 1
        return ASRResult(
            segments=[
                Segment(
                    start=0.0,
                    end=1.8,
                    speaker_id=None,
                    raw_text="Привет, си мейк и ви ти кей.",
                ),
                Segment(
                    start=2.0,
                    end=3.5,
                    speaker_id=None,
                    raw_text="Обсудим докер.",
                ),
            ],
            language="ru",
            backend="fake-whisper",
            model="fake-asr",
            words_available=False,
        )

    def release(self):
        return None


class FakeDiarizer:
    calls = 0

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def diarize(self, wav_path, num_speakers=None):
        FakeDiarizer.calls += 1
        return DiarizationResult(
            intervals=[
                DiarizationInterval(0.0, 1.9, "speaker_00"),
                DiarizationInterval(2.0, 3.6, "speaker_00"),
            ],
            backend="fake-nemo",
            model="fake-titanet",
        )

    def release(self):
        return None


@pytest.fixture()
def cfg(tmp_path):
    config = load_config(ROOT / "config/default.yaml")
    config.paths.models_dir = str(tmp_path / "models")
    config.paths.work_dir = str(tmp_path / "work")
    config.paths.output_dir = str(tmp_path / "output")
    config.terminology.profiles = ["general-it"]
    config.llm.enabled = False
    return config


def test_full_pipeline_with_fake_backends(monkeypatch, tmp_path, cfg):
    FakeWhisper.calls = 0
    FakeDiarizer.calls = 0

    def fake_preprocess(input_path, work_dir, audio_cfg, chunk_seconds=900, overlap_seconds=15):
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        wav = work / "audio_16k_mono.wav"
        wav.write_bytes(b"fake-wav")
        return PreparedAudio(
            source=Path(input_path),
            wav_path=wav,
            metadata=AudioMetadata(
                filename=Path(input_path).name,
                path=str(input_path),
                duration=4.0,
                sample_rate=16000,
                channels=1,
                sha256="deadbeef",
            ),
            chunks=[AudioChunk(0, 0.0, 4.0, wav)],
        )

    monkeypatch.setattr(runner, "preprocess_audio", fake_preprocess)
    monkeypatch.setattr(runner, "WhisperBackend", FakeWhisper)
    monkeypatch.setattr(runner, "NemoDiarizer", FakeDiarizer)

    input_audio = tmp_path / "meeting.wav"
    input_audio.write_bytes(b"source")

    result = runner.Pipeline(cfg).run(input_audio)
    transcript = result.transcript
    assert FakeWhisper.calls == 1
    assert FakeDiarizer.calls == 1
    assert len(transcript.segments) == 2
    assert all(s.speaker_id == "speaker_00" for s in transcript.segments)
    assert "CMake" in transcript.segments[0].corrected_text
    assert "VTK" in transcript.segments[0].corrected_text
    assert "Docker" in transcript.segments[1].corrected_text
    assert transcript.segments[0].raw_text == "Привет, си мейк и ви ти кей."

    assert set(result.outputs) == {"json", "markdown", "srt"}
    for path in result.outputs.values():
        assert path.is_file()
    assert "speaker_00" in result.outputs["srt"].read_text(encoding="utf-8")

    # Re-run resumes completed stages instead of re-running models.
    result2 = runner.Pipeline(cfg).run(input_audio)
    assert FakeWhisper.calls == 1
    assert FakeDiarizer.calls == 1
    assert len(result2.transcript.segments) == 2

    # Configuration changes invalidate only the affected stage (terminology),
    # so ASR is not re-run.
    cfg.terminology.threshold = 0.95
    runner.Pipeline(cfg).run(input_audio)
    assert FakeWhisper.calls == 1


def test_skip_diarization_marks_speakers_unknown(monkeypatch, tmp_path, cfg):
    def fake_preprocess(input_path, work_dir, audio_cfg, chunk_seconds=900, overlap_seconds=15):
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        wav = work / "audio_16k_mono.wav"
        wav.write_bytes(b"fake-wav")
        return PreparedAudio(
            source=Path(input_path),
            wav_path=wav,
            metadata=AudioMetadata(filename="x.wav", duration=2.0, sample_rate=16000, channels=1),
            chunks=[AudioChunk(0, 0.0, 2.0, wav)],
        )

    monkeypatch.setattr(runner, "preprocess_audio", fake_preprocess)
    monkeypatch.setattr(runner, "WhisperBackend", FakeWhisper)

    input_audio = tmp_path / "meeting.wav"
    input_audio.write_bytes(b"source")
    result = runner.Pipeline(cfg, skip_diarization=True).run(input_audio)
    assert result.transcript.speaker_ids == []
    assert any("diarization" in w for w in result.transcript.warnings)


def test_enabling_diarization_after_skip_invalidates_cache(monkeypatch, tmp_path, cfg):
    FakeDiarizer.calls = 0

    def fake_preprocess(input_path, work_dir, audio_cfg, chunk_seconds=900, overlap_seconds=15):
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        wav = work / "audio_16k_mono.wav"
        wav.write_bytes(b"fake-wav")
        return PreparedAudio(
            source=Path(input_path),
            wav_path=wav,
            metadata=AudioMetadata(filename="x.wav", duration=4.0, sample_rate=16000, channels=1),
            chunks=[AudioChunk(0, 0.0, 4.0, wav)],
        )

    monkeypatch.setattr(runner, "preprocess_audio", fake_preprocess)
    monkeypatch.setattr(runner, "WhisperBackend", FakeWhisper)
    monkeypatch.setattr(runner, "NemoDiarizer", FakeDiarizer)

    input_audio = tmp_path / "meeting.wav"
    input_audio.write_bytes(b"source")

    first = runner.Pipeline(cfg, skip_diarization=True).run(input_audio)
    assert first.transcript.speaker_ids == []

    second = runner.Pipeline(cfg, skip_diarization=False).run(input_audio)
    assert FakeDiarizer.calls == 1
    assert second.transcript.speaker_ids == ["speaker_00"]
