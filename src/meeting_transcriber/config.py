"""Configuration loading and validation.

Configuration is YAML with a fixed, typed schema. Unknown keys are rejected so
typos do not silently change pipeline behaviour. Relative paths are resolved
against ``paths.root`` (the repository root by default), never against the
current working directory.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError

_ENV_PATTERN = re.compile(r"\{env:([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass
class PathsConfig:
    root: str = "."
    models_dir: str = "models"
    work_dir: str = "work"
    output_dir: str = "data/output"
    terminology_dir: str = "config/terminology"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    # When true, a video input's selected audio track is copied losslessly to
    # work/<meeting>/audio_original.mka before conversion.
    keep_original: bool = True
    # 0-based index among audio streams (FFmpeg 0:a:N) for multi-track videos.
    audio_stream: int = 0


@dataclass
class ASRConfig:
    backend: str = "whisper"
    model_dir: str = "whisper-large-v3-turbo-ct2"
    device: str = "cuda"
    compute_type: str = "float16"
    cpu_threads: int = 4
    beam_size: int = 5
    word_timestamps: bool = True
    vad_filter: bool = True
    vad_min_silence_ms: int = 500
    condition_on_previous_text: bool = False
    initial_prompt: str = (
        "Совещание на русском языке с английскими техническими терминами: "
        "CMake, Qt, VTK, API, GPU, Python, C++, Git, Docker."
    )
    chunk_seconds: float = 900.0
    overlap_seconds: float = 15.0
    local_files_only: bool = True


@dataclass
class DiarizationConfig:
    enabled: bool = True
    backend: str = "nemo"
    device: str = "cuda"
    speaker_model: str = "speakerverification_en_titanet_large.nemo"
    max_speakers: int = 20
    num_speakers: int | None = None
    # Silero VAD is used to build the external VAD manifest for NeMo.
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 500
    vad_speech_pad_ms: int = 100
    window_lengths: list[float] = field(
        default_factory=lambda: [1.5, 1.25, 1.0, 0.75, 0.5]
    )
    shift_lengths: list[float] = field(
        default_factory=lambda: [0.75, 0.625, 0.5, 0.375, 0.1]
    )
    multiscale_weights: list[float] = field(default_factory=lambda: [1, 1, 1, 1, 1])
    enhanced_count_thres: int = 80
    max_rp_threshold: float = 0.25
    sparse_search_volume: int = 30
    # Very long recordings produce tens of thousands of embedding windows, and
    # NeMo's spectral clustering builds an O(N^2) matrix. Above this duration we
    # switch to a single-scale embedding grid (see long_audio_window/shift).
    long_audio_threshold_seconds: int = 1800
    long_audio_window: float = 1.5
    long_audio_shift: float = 0.75
    keep_intermediate: bool = False


@dataclass
class TerminologyConfig:
    enabled: bool = True
    profiles: list[str] = field(default_factory=lambda: ["general-it"])
    glossary_files: list[str] = field(default_factory=list)
    threshold: float = 0.86
    apply_corrections: bool = True
    fuzzy: bool = True
    phonetic: bool = False
    max_ngram: int = 5
    ambiguous_margin: float = 0.04


@dataclass
class LLMConfig:
    enabled: bool = True
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "qwen3.5-9b"
    api_key_env: str = "OPENAI_API_KEY"
    timeout: int = 600
    temperature: float = 0.1
    max_tokens: int = 4096
    chunk_chars: int = 5000
    merge_chars: int = 12000
    max_topics: int = 12
    num_ctx: int = 8192
    retries: int = 1


@dataclass
class OutputConfig:
    formats: list[str] = field(default_factory=lambda: ["json", "markdown", "srt"])
    write_words: bool = True
    overwrite: bool = False


@dataclass
class PipelineConfig:
    resume: bool = True
    continue_on_diarization_error: bool = True
    keep_work_dir: bool = True


@dataclass
class AppConfig:
    language: str = "ru"
    meeting_id: str | None = None
    paths: PathsConfig = field(default_factory=PathsConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    terminology: TerminologyConfig = field(default_factory=TerminologyConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    @property
    def root(self) -> Path:
        return Path(self.paths.root).expanduser().resolve()

    def resolve(self, path: str | Path) -> Path:
        p = Path(path).expanduser()
        return p if p.is_absolute() else (self.root / p).resolve()

    def models_path(self, *parts: str) -> Path:
        return self.resolve(self.paths.models_dir).joinpath(*parts)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)


def _to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return {f.name: _to_plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    return value


def _substitute_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env(v) for v in value]
    return value


def _build(cls: type, data: dict[str, Any], prefix: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{prefix or 'config'}: expected a mapping, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(f"{prefix or 'config'}: unknown key(s): {', '.join(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict) and is_dataclass(f.type):
            kwargs[name] = _build(f.type, value, key)  # pragma: no cover - nested config
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _defaults() -> dict[str, Any]:
    return _to_plain(AppConfig())


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> AppConfig:
    """Load YAML config merged on top of built-in defaults."""

    data = _defaults()
    if path is not None:
        config_path = Path(path).expanduser()
        if not config_path.is_file():
            raise ConfigError(f"configuration file not found: {config_path}")
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ConfigError(f"{config_path}: top-level YAML document must be a mapping")
        data = _deep_merge(data, loaded)
    if overrides:
        data = _deep_merge(data, overrides)
    data = _substitute_env(data)

    cfg = AppConfig(
        language=data["language"],
        meeting_id=data.get("meeting_id"),
        paths=_build(PathsConfig, data["paths"], "paths"),
        audio=_build(AudioConfig, data["audio"], "audio"),
        asr=_build(ASRConfig, data["asr"], "asr"),
        diarization=_build(DiarizationConfig, data["diarization"], "diarization"),
        terminology=_build(TerminologyConfig, data["terminology"], "terminology"),
        llm=_build(LLMConfig, data["llm"], "llm"),
        output=_build(OutputConfig, data["output"], "output"),
        pipeline=_build(PipelineConfig, data["pipeline"], "pipeline"),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: AppConfig) -> None:
    if cfg.audio.sample_rate != 16000:
        raise ConfigError(
            "audio.sample_rate must be 16000: both Whisper and NeMo diarization "
            "expect 16 kHz mono audio"
        )
    if cfg.audio.channels != 1:
        raise ConfigError("audio.channels must be 1 (pipeline creates a mono derivative)")
    if cfg.audio.audio_stream < 0:
        raise ConfigError("audio.audio_stream must be >= 0")
    if cfg.asr.device not in {"cuda", "cpu", "auto"}:
        raise ConfigError("asr.device must be one of: cuda, cpu, auto")
    if cfg.asr.chunk_seconds <= 0:
        raise ConfigError("asr.chunk_seconds must be positive")
    if cfg.asr.overlap_seconds < 0 or cfg.asr.overlap_seconds * 2 >= cfg.asr.chunk_seconds:
        raise ConfigError("asr.overlap_seconds must be >= 0 and less than half of chunk_seconds")
    if cfg.diarization.max_speakers < 1:
        raise ConfigError("diarization.max_speakers must be >= 1")
    if cfg.diarization.num_speakers is not None:
        if cfg.diarization.num_speakers < 1:
            raise ConfigError("diarization.num_speakers must be >= 1")
        if cfg.diarization.num_speakers > cfg.diarization.max_speakers:
            raise ConfigError("diarization.num_speakers cannot exceed diarization.max_speakers")
    lengths = (
        len(cfg.diarization.window_lengths),
        len(cfg.diarization.shift_lengths),
        len(cfg.diarization.multiscale_weights),
    )
    if len(set(lengths)) != 1:
        raise ConfigError(
            "diarization.window_lengths, shift_lengths and multiscale_weights "
            "must have the same length"
        )
    if not 0.0 <= cfg.terminology.threshold <= 1.0:
        raise ConfigError("terminology.threshold must be within [0, 1]")
    unknown_formats = set(cfg.output.formats) - {"json", "markdown", "srt"}
    if unknown_formats:
        raise ConfigError(f"unsupported output.formats: {sorted(unknown_formats)}")
    if cfg.diarization.long_audio_threshold_seconds < 0:
        raise ConfigError("diarization.long_audio_threshold_seconds must be >= 0")
    if cfg.diarization.long_audio_window <= cfg.diarization.long_audio_shift:
        raise ConfigError("diarization.long_audio_window must be greater than long_audio_shift")
    if cfg.diarization.long_audio_shift <= 0:
        raise ConfigError("diarization.long_audio_shift must be positive")
    if cfg.llm.chunk_chars <= 0:
        raise ConfigError("llm.chunk_chars must be positive")
    if cfg.llm.merge_chars <= 0:
        raise ConfigError("llm.merge_chars must be positive")
    if cfg.llm.retries < 0:
        raise ConfigError("llm.retries must be >= 0")
    if cfg.llm.max_topics < 1:
        raise ConfigError("llm.max_topics must be >= 1")
    if cfg.llm.num_ctx < 2048:
        raise ConfigError("llm.num_ctx must be >= 2048")
