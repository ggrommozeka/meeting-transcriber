from pathlib import Path

import pytest

from meeting_transcriber.config import load_config
from meeting_transcriber.errors import ConfigError

ROOT = Path(__file__).resolve().parents[2]


def test_default_config_is_valid():
    cfg = load_config(ROOT / "config/default.yaml")
    assert cfg.language == "ru"
    assert cfg.audio.sample_rate == 16000
    assert cfg.diarization.max_speakers == 20
    assert cfg.asr.local_files_only is True
    assert cfg.llm.enabled is True
    assert cfg.llm.model == "qwen3.5-9b"
    assert cfg.models_path().name == "models"


def test_docker_config_is_valid():
    cfg = load_config(ROOT / "config/docker.yaml")
    assert cfg.paths.models_dir == "/models"
    assert cfg.paths.output_dir == "/data/output"


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("asr:\n  typo_option: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(path)


def test_invalid_overlap_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("asr:\n  chunk_seconds: 10\n  overlap_seconds: 5\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="overlap_seconds"):
        load_config(path)


def test_num_speakers_cannot_exceed_max(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("diarization:\n  num_speakers: 30\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="cannot exceed"):
        load_config(path)


def test_negative_audio_stream_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("audio:\n  audio_stream: -1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="audio_stream"):
        load_config(path)


def test_invalid_long_audio_scale_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "diarization:\n  long_audio_window: 0.5\n  long_audio_shift: 0.75\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="long_audio_window"):
        load_config(path)


def test_env_substitution(tmp_path, monkeypatch):
    monkeypatch.setenv("MT_TEST_MODEL_DIR", "custom-dir")
    path = tmp_path / "env.yaml"
    path.write_text("asr:\n  model_dir: '{env:MT_TEST_MODEL_DIR}'\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.asr.model_dir == "custom-dir"


def test_overrides_are_merged():
    cfg = load_config(ROOT / "config/default.yaml", {"paths": {"models_dir": "/srv/models"}})
    assert cfg.paths.models_dir == "/srv/models"
    assert cfg.asr.device == "cuda"
