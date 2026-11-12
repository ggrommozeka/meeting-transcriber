from pathlib import Path

import pytest

from meeting_transcriber.config import load_config
from meeting_transcriber.errors import ConfigError
from meeting_transcriber.setup.config_store import load_local_config, save_local_settings
from meeting_transcriber.setup.diagnostics import Check, DiagnosticsReport
from meeting_transcriber.setup.llama_cpp import resolve_llama_binary

ROOT = Path(__file__).resolve().parents[2]


def test_readiness_weighted_score():
    report = DiagnosticsReport(
        checks=[
            Check("a", "A", "ok", 100.0, weight=1),
            Check("b", "B", "fail", 0.0, weight=1),
            Check("c", "C", "skipped", 100.0, weight=2),
        ]
    )
    assert report.overall == pytest.approx(75.0)
    assert [check.key for check in report.blocking] == ["b"]


def test_local_config_roundtrip(tmp_path):
    local = tmp_path / "local.yaml"
    save_local_settings(
        {
            "runtime": {"llama_server": "/opt/llama/llama-server"},
            "visual": {"enabled": True},
        },
        local,
    )
    cfg = load_local_config(ROOT / "config/default.yaml", local)
    assert cfg.runtime.llama_server == "/opt/llama/llama-server"
    assert cfg.visual.enabled is True


def test_resolve_llama_binary_prefers_existing_path(tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"#!/bin/sh\n")
    cfg = load_config(ROOT / "config/default.yaml")
    cfg.runtime.llama_server = str(binary)
    assert resolve_llama_binary(cfg) == binary


def test_invalid_runtime_source_is_rejected():
    cfg = load_config(ROOT / "config/default.yaml")
    cfg.runtime.llama_source = "magic"
    with pytest.raises(ConfigError, match="llama_source"):
        from meeting_transcriber.config import validate_config

        validate_config(cfg)
