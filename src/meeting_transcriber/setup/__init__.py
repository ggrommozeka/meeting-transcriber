"""Setup helpers: diagnostics, llama.cpp management and local settings."""

from .config_store import load_local_config, save_local_settings
from .diagnostics import Check, DiagnosticsReport, run_diagnostics
from .llama_cpp import (
    LlamaError,
    build_from_source,
    download_release,
    install_llama,
    is_server_running,
    latest_tag,
    resolve_llama,
    resolve_llama_binary,
)

__all__ = [
    "Check",
    "DiagnosticsReport",
    "LlamaError",
    "build_from_source",
    "download_release",
    "install_llama",
    "is_server_running",
    "latest_tag",
    "load_local_config",
    "resolve_llama",
    "resolve_llama_binary",
    "run_diagnostics",
    "save_local_settings",
]
