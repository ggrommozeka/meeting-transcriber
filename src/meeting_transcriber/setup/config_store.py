"""Local configuration overlay (``config/local.yaml``).

The tracked ``config/default.yaml`` stays untouched; machine-specific paths and
wizard choices are written to an ignored local overlay.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..config import AppConfig, load_config

LOCAL_CONFIG = "config/local.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    return data


def load_local_config(
    base_path: str | Path = "config/default.yaml",
    local_path: str | Path = LOCAL_CONFIG,
) -> AppConfig:
    """Load the tracked config and merge ``config/local.yaml`` on top."""

    base = Path(base_path).expanduser()
    local = Path(local_path).expanduser()
    overrides = _read_yaml(local)
    if base.is_file():
        return load_config(base, overrides=overrides or None)
    return load_config(None, overrides=overrides or None)


def save_local_settings(
    data: dict[str, Any],
    path: str | Path = LOCAL_CONFIG,
) -> Path:
    """Atomically merge and write local settings."""

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    merged = _deep_merge(_read_yaml(target), data)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(
        yaml.safe_dump(merged, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    tmp.replace(target)
    return target


def config_values(cfg: AppConfig) -> dict[str, Any]:
    """Minimal local overlay representing wizard choices."""

    return {
        "paths": {
            "models_dir": cfg.paths.models_dir,
            "work_dir": cfg.paths.work_dir,
            "output_dir": cfg.paths.output_dir,
        },
        "audio": {"ffmpeg": cfg.audio.ffmpeg, "ffprobe": cfg.audio.ffprobe},
        "visual": {
            "enabled": cfg.visual.enabled,
            "tesseract": cfg.visual.tesseract,
            "tessdata_dir": cfg.visual.tessdata_dir,
        },
        "runtime": {
            "llama_server": cfg.runtime.llama_server,
            "llama_source": cfg.runtime.llama_source,
            "llama_tag": cfg.runtime.llama_tag,
            "llama_build_dir": cfg.runtime.llama_build_dir,
            "llama_cuda": cfg.runtime.llama_cuda,
            "llama_jobs": cfg.runtime.llama_jobs,
        },
        "llm": {"enabled": cfg.llm.enabled},
    }


__all__ = [
    "LOCAL_CONFIG",
    "config_values",
    "load_local_config",
    "save_local_settings",
]
