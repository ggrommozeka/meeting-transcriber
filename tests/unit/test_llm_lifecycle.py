from pathlib import Path

import pytest

from meeting_transcriber.config import load_config
from meeting_transcriber.pipeline import runner
from meeting_transcriber.pipeline.runner import Pipeline

ROOT = Path(__file__).resolve().parents[2]


class FakeServer:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __enter__(self):
        self.events.append("start")
        return None

    def __exit__(self, *exc):
        self.events.append("stop")


def _pipeline(tmp_path: Path, events: list[str], **kwargs) -> Pipeline:
    cfg = load_config(ROOT / "config/default.yaml")
    cfg.llm.enabled = True
    return Pipeline(
        cfg,
        force=kwargs.pop("force", True),
        report_server=lambda: FakeServer(events),
        **kwargs,
    )


def test_report_server_starts_only_for_report(tmp_path, monkeypatch):
    events: list[str] = []
    pipeline = _pipeline(tmp_path, events)

    def fake_report(*args, **kwargs):
        assert events == ["start"], "llama-server must start immediately before the report"
        return {"summary": "ok"}, []

    monkeypatch.setattr(runner, "generate_meeting_report", fake_report)
    report, warnings = pipeline._stage_report(object(), [], tmp_path, {}, "fp")

    assert report == {"summary": "ok"}
    assert warnings == []
    assert events == ["start", "stop"]
    assert (tmp_path / "report.json").is_file()


def test_report_server_not_started_for_cached_report(tmp_path):
    events: list[str] = []
    pipeline = _pipeline(tmp_path, events, force=False)
    artifact = tmp_path / "report.json"
    artifact.write_text('{"summary": "cached"}', encoding="utf-8")
    state = {
        "stages": {
            "llm_report": {
                "completed": True,
                "artifact": str(artifact),
                "fingerprint": "fp",
            }
        }
    }

    report, _ = pipeline._stage_report(object(), [], tmp_path, state, "fp")

    assert report == {"summary": "cached"}
    assert events == []


def test_report_server_not_started_when_llm_is_skipped(tmp_path):
    events: list[str] = []
    pipeline = _pipeline(tmp_path, events, skip_llm=True)

    assert pipeline._stage_report(object(), [], tmp_path, {}, "fp") == (None, [])
    assert events == []


def test_stale_llama_server_is_stopped_before_asr(tmp_path, monkeypatch):
    events: list[str] = []
    pipeline = _pipeline(tmp_path, events)
    calls: list[Path] = []

    monkeypatch.setattr(
        "meeting_transcriber.setup.llama_cpp.is_server_running", lambda url: True
    )

    def fake_stop(cfg, pid_path=None):
        calls.append(pid_path)
        return True

    monkeypatch.setattr(
        "meeting_transcriber.setup.llama_cpp.stop_managed_server", fake_stop
    )

    assert pipeline._stop_stale_llm_server() is True
    assert calls == [pipeline.cfg.resolve("logs/llama-server.pid")]


def test_unstoppable_llama_server_is_an_error(tmp_path, monkeypatch):
    events: list[str] = []
    pipeline = _pipeline(tmp_path, events)
    monkeypatch.setattr(
        "meeting_transcriber.setup.llama_cpp.is_server_running", lambda url: True
    )
    monkeypatch.setattr(
        "meeting_transcriber.setup.llama_cpp.stop_managed_server",
        lambda cfg, pid_path=None: False,
    )

    with pytest.raises(runner.PipelineError, match="--skip-llm"):
        pipeline._stop_stale_llm_server()
