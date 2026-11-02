"""Typed errors and process exit codes."""

from __future__ import annotations

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2
EXIT_MODEL_MISSING = 3
EXIT_GPU = 4


class TranscriberError(Exception):
    """Base error for the pipeline."""

    exit_code = EXIT_ERROR


class ConfigError(TranscriberError):
    exit_code = EXIT_CONFIG


class ModelNotFoundError(TranscriberError):
    exit_code = EXIT_MODEL_MISSING


class ModelIntegrityError(TranscriberError):
    exit_code = EXIT_MODEL_MISSING


class GPUUnavailableError(TranscriberError):
    exit_code = EXIT_GPU


class AudioError(TranscriberError):
    pass


class BackendUnavailableError(TranscriberError):
    """Raised when an optional backend dependency is not installed."""


class PipelineError(TranscriberError):
    pass
