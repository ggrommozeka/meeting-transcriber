"""Environment diagnostics and readiness scoring for setup/process screens."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ..config import AppConfig
from ..models import iter_groups, load_manifest


@dataclass
class Check:
    key: str
    label: str
    status: str
    progress: float
    detail: str = ""
    weight: float = 1.0
    action: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in {"ok", "skipped"}

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "label": self.label,
            "status": self.status,
            "progress": round(self.progress, 1),
            "detail": self.detail,
            "weight": self.weight,
            "action": self.action,
        }


@dataclass
class DiagnosticsReport:
    checks: list[Check] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def overall(self) -> float:
        total = sum(check.weight for check in self.checks)
        if total <= 0:
            return 100.0
        return round(
            sum(check.weight * check.progress for check in self.checks) / total, 1
        )

    @property
    def blocking(self) -> list[Check]:
        return [check for check in self.checks if check.status == "fail"]

    def to_dict(self) -> dict[str, object]:
        return {
            "overall": self.overall,
            "checks": [check.to_dict() for check in self.checks],
        }


def _module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _check(
    key: str,
    label: str,
    ok: bool,
    weight: float,
    detail: str,
    *,
    skipped: bool = False,
    action: str | None = None,
    warn: bool = False,
) -> Check:
    if skipped:
        return Check(key, label, "skipped", 100.0, "не требуется", weight)
    if ok:
        return Check(key, label, "ok", 100.0, detail, weight)
    return Check(
        key,
        label,
        "warn" if warn else "fail",
        0.0 if not warn else 50.0,
        detail,
        weight,
        action,
    )


def _resolve_binary(value: str | None, fallback: str | None = None) -> str | None:
    if value:
        path = Path(value).expanduser()
        if path.is_file():
            return str(path)
        found = shutil.which(value)
        return found
    return shutil.which(fallback) if fallback else None


def _tesseract_languages(binary: str, tessdata_dir: str | None) -> tuple[bool, str]:
    env = None
    if tessdata_dir:
        import os

        env = os.environ.copy()
        env["TESSDATA_PREFIX"] = str(Path(tessdata_dir).expanduser())
    try:
        completed = subprocess.run(
            [binary, "--list-langs"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if completed.returncode != 0:
        return False, completed.stderr.strip() or "tesseract failed"
    languages = {
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip() and "available languages" not in line
    }
    missing = {"rus", "eng"} - languages
    if missing:
        return False, "нет языков: " + ", ".join(sorted(missing))
    return True, "rus, eng"


def _llama_check(binary: str | None) -> tuple[bool, str]:
    if not binary:
        return False, "llama-server не выбран"
    path = Path(binary).expanduser()
    if not path.is_file():
        return False, f"файл не найден: {path}"
    try:
        completed = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    text = (completed.stdout + completed.stderr).strip().splitlines()
    return completed.returncode == 0, text[0] if text else str(path)


def _nvidia() -> tuple[bool, str]:
    binary = shutil.which("nvidia-smi")
    if not binary:
        return False, "nvidia-smi не найден"
    try:
        completed = subprocess.run(
            [
                binary,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    if completed.returncode != 0:
        return False, completed.stderr.strip() or "nvidia-smi failed"
    return True, completed.stdout.strip().splitlines()[0]


def run_diagnostics(
    cfg: AppConfig,
    *,
    visual: bool | None = None,
    diarization: bool | None = None,
    report: bool | None = None,
    llama_server: str | None = None,
    deep_models: bool = False,
) -> DiagnosticsReport:
    """Check the environment for the selected workflow."""

    visual = cfg.visual.enabled if visual is None else visual
    diarization = cfg.diarization.enabled if diarization is None else diarization
    report = cfg.llm.enabled if report is None else report
    models_dir = cfg.models_path()
    report_obj = DiagnosticsReport()

    python_ok = sys.version_info >= (3, 10)
    report_obj.checks.append(
        _check(
            "python",
            "Python",
            python_ok,
            5,
            sys.version.split()[0],
        )
    )

    core = ["yaml", "numpy", "requests", "PIL", "rich"]
    missing_core = [name for name in core if not _module(name)]
    report_obj.checks.append(
        _check(
            "python-core",
            "Python-зависимости",
            not missing_core,
            10,
            "base OK" if not missing_core else "нет: " + ", ".join(missing_core),
            action="pip install -r requirements/base.txt",
        )
    )

    asr_missing = [name for name in ("faster_whisper", "av") if not _module(name)]
    report_obj.checks.append(
        _check(
            "asr",
            "ASR-зависимости",
            not asr_missing,
            10,
            "faster-whisper, av" if not asr_missing else "нет: " + ", ".join(asr_missing),
            action="pip install -r requirements/asr.txt",
        )
    )

    if diarization:
        diar_missing = [
            name
            for name in ("torch", "nemo", "silero_vad")
            if not _module(name)
        ]
        report_obj.checks.append(
            _check(
                "diarization-deps",
                "Диаризация",
                not diar_missing,
                10,
                "torch, NeMo, Silero"
                if not diar_missing
                else "нет: " + ", ".join(diar_missing),
                action="pip install -r requirements/diarization.txt",
            )
        )
    else:
        report_obj.checks.append(
            _check("diarization-deps", "Диаризация", True, 10, "", skipped=True)
        )

    ffmpeg = _resolve_binary(cfg.audio.ffmpeg, "ffmpeg")
    ffprobe = _resolve_binary(cfg.audio.ffprobe, "ffprobe")
    report_obj.checks.append(
        _check(
            "ffmpeg",
            "FFmpeg / ffprobe",
            bool(ffmpeg and ffprobe),
            10,
            f"{ffmpeg or '—'} / {ffprobe or '—'}",
            action="scripts/install_host_dependencies.sh",
        )
    )

    if visual:
        binary = _resolve_binary(cfg.visual.tesseract, "tesseract")
        ok, detail = (
            _tesseract_languages(binary, cfg.visual.tessdata_dir)
            if binary
            else (False, "tesseract не найден")
        )
        report_obj.checks.append(
            _check(
                "visual",
                "Tesseract OCR",
                ok,
                10,
                detail,
                action="scripts/install_host_dependencies.sh",
            )
        )
    else:
        report_obj.checks.append(
            _check("visual", "Tesseract OCR", True, 10, "", skipped=True)
        )

    resolved_llama = llama_server or cfg.runtime.llama_server
    if report and not resolved_llama:
        from .llama_cpp import resolve_llama_binary

        detected = resolve_llama_binary(cfg)
        resolved_llama = str(detected) if detected else None
    if report:
        ok, detail = _llama_check(resolved_llama)
        report_obj.checks.append(
            _check(
                "llama",
                "llama-server",
                ok,
                10,
                detail,
                action="python -m meeting_transcriber.cli setup",
            )
        )
    else:
        report_obj.checks.append(
            _check("llama", "llama-server", True, 10, "", skipped=True)
        )

    groups_needed = {"asr"}
    if diarization:
        groups_needed.add("diarization")
    if report:
        groups_needed.add("llm")
    try:
        manifest_path = cfg.resolve("models/manifest.yaml")
        if not manifest_path.is_file():
            manifest_path = cfg.models_path("manifest.yaml")
        manifest = load_manifest(manifest_path)
        groups = {group.key: group for group in iter_groups(manifest)}
        for key, label in (
            ("asr", "Модель ASR"),
            ("diarization", "Модель диаризации"),
            ("llm", "Модель LLM"),
        ):
            group = groups.get(key)
            if key not in groups_needed:
                report_obj.checks.append(
                    _check(key + "-model", label, True, 10, "", skipped=True)
                )
                continue
            if group is None:
                report_obj.checks.append(
                    _check(key + "-model", label, False, 10, "группа не найдена")
                )
                continue
            missing = [
                item.name
                for item in group.files
                if not (models_dir / group.target_dir / item.name).is_file()
            ]
            detail = (
                f"{len(group.files)} файл(ов) на месте"
                if not missing
                else "нет: " + ", ".join(missing[:3])
            )
            report_obj.checks.append(
                _check(
                    key + "-model",
                    label,
                    not missing,
                    10,
                    detail,
                    action="download-models --only " + key,
                )
            )
    except Exception as exc:  # diagnostics must never crash the UI
        report_obj.checks.append(
            _check("models", "Модели", False, 30, f"manifest: {exc}")
        )

    gpu_ok, gpu_detail = _nvidia()
    report_obj.checks.append(
        _check(
            "gpu",
            "GPU / драйвер",
            gpu_ok,
            5,
            gpu_detail,
            warn=cfg.asr.device != "cuda",
        )
    )

    try:
        usage = shutil.disk_usage(cfg.root)
        free_gib = usage.free / 1024**3
        report_obj.checks.append(
            _check(
                "disk",
                "Свободное место",
                free_gib >= 5,
                5,
                f"{free_gib:.1f} GiB",
                warn=free_gib >= 2,
            )
        )
    except OSError as exc:
        report_obj.checks.append(_check("disk", "Свободное место", False, 5, str(exc)))

    return report_obj


__all__ = ["Check", "DiagnosticsReport", "run_diagnostics"]
