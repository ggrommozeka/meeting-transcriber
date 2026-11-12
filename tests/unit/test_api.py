from pathlib import Path

from meeting_transcriber import api
from meeting_transcriber.pipeline.runner import RunResult
from meeting_transcriber.types import AudioMetadata, Transcript

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/default.yaml"


class FakePipeline:
    calls = []

    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.kwargs = kwargs
        FakePipeline.calls.append(self)

    def run(self, source):
        return RunResult(
            transcript=Transcript(
                meeting_id="m",
                language="ru",
                source_audio=AudioMetadata(filename="m.mp4"),
            )
        )


def test_two_api_functions_use_sensible_defaults(monkeypatch, tmp_path):
    FakePipeline.calls = []
    monkeypatch.setattr(api, "Pipeline", FakePipeline)
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"video")

    api.transcribe_audio(source, config=CONFIG)
    audio = FakePipeline.calls[-1]
    assert audio.cfg.visual.enabled is False
    assert audio.kwargs["skip_llm"] is True
    assert Path(audio.cfg.paths.output_dir) == tmp_path

    api.transcribe_video(source, config=CONFIG)
    video = FakePipeline.calls[-1]
    assert video.cfg.visual.enabled is True
    assert video.cfg.visual.keep_code_frames is True
    assert video.kwargs["skip_llm"] is True
    assert Path(video.cfg.paths.output_dir) == tmp_path

    api.transcribe_audio(source, output_dir=tmp_path / "out", config=CONFIG)
    assert Path(FakePipeline.calls[-1].cfg.paths.output_dir) == tmp_path / "out"
