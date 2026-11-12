"""Keyboard-first setup wizard built with questionary."""

from __future__ import annotations

import questionary
from questionary import Choice

from ..config import AppConfig
from ..models import download_models, iter_groups, load_manifest
from ..setup.config_store import config_values, load_local_config, save_local_settings
from ..setup.diagnostics import run_diagnostics
from ..setup.llama_cpp import LlamaError, install_llama
from ..ui import DownloadProgress, banner, console
from .common import STYLE, ask_path, choose_from_options, print_readiness


def _ask_text(message: str, default: str, validate=None) -> str | None:
    return questionary.text(
        message,
        default=default,
        style=STYLE,
        validate=validate,
    ).ask()


def _ask_models_dir(cfg: AppConfig) -> str:
    value = questionary.text(
        "Каталог моделей",
        default=cfg.paths.models_dir,
        style=STYLE,
        validate=lambda text: True if text.strip() else "Укажите каталог",
    ).ask()
    return value.strip() if value else cfg.paths.models_dir


def _ask_model_groups(cfg: AppConfig) -> list[str]:
    manifest_path = cfg.resolve("models/manifest.yaml")
    if not manifest_path.is_file():
        manifest_path = cfg.models_path("manifest.yaml")
    choices = []
    try:
        manifest = load_manifest(manifest_path)
        for group in iter_groups(manifest):
            present = all(
                (cfg.models_path() / group.target_dir / item.name).is_file()
                for item in group.files
            )
            choices.append(
                Choice(
                    f"{group.key} — {group.name}" + (" (есть)" if present else ""),
                    value=group.key,
                    checked=present,
                )
            )
    except Exception:
        choices = [
            Choice("asr", value="asr", checked=True),
            Choice("diarization", value="diarization", checked=True),
            Choice("llm", value="llm"),
        ]
    selected = questionary.checkbox(
        "Какие группы моделей нужны?",
        choices=choices,
        style=STYLE,
        instruction="Space — переключить · Enter — далее",
    ).ask()
    return selected or []


def _ask_llama(cfg: AppConfig) -> None:
    source = choose_from_options(
        "Источник llama.cpp",
        [
            Choice("Готовый бинарник", value="existing"),
            Choice("Скачать готовый релиз", value="release"),
            Choice("Собрать из исходников", value="build"),
        ],
    )
    if source is None:
        return
    cfg.runtime.llama_source = str(source)
    if source == "existing":
        path = ask_path(
            "Путь к llama-server",
            default=cfg.runtime.llama_server or "",
            media=False,
        )
        if path is not None:
            cfg.runtime.llama_server = str(path)
        return
    tag = _ask_text("Тег llama.cpp (latest = последний стабильный)", cfg.runtime.llama_tag)
    if tag:
        cfg.runtime.llama_tag = tag
    build_dir = _ask_text("Каталог сборки/установки", cfg.runtime.llama_build_dir)
    if build_dir:
        cfg.runtime.llama_build_dir = build_dir
    cuda = questionary.confirm(
        "Собирать/скачивать с CUDA?", default=cfg.runtime.llama_cuda, style=STYLE
    ).ask()
    if cuda is not None:
        cfg.runtime.llama_cuda = cuda
    jobs = _ask_text("Параллельных jobs (0 = авто)", str(cfg.runtime.llama_jobs))
    if jobs and jobs.strip().isdigit():
        cfg.runtime.llama_jobs = int(jobs)


def _collect(cfg: AppConfig) -> tuple[AppConfig, list[str]]:
    cfg.paths.models_dir = _ask_models_dir(cfg)
    visual = questionary.confirm(
        "Включить OCR видео?", default=cfg.visual.enabled, style=STYLE
    ).ask()
    if visual is not None:
        cfg.visual.enabled = visual
    tesseract = _ask_text("Путь к tesseract", cfg.visual.tesseract)
    if tesseract:
        cfg.visual.tesseract = tesseract
    tessdata = _ask_text("Каталог tessdata (пусто = системный)", cfg.visual.tessdata_dir or "")
    if tessdata is not None:
        cfg.visual.tessdata_dir = tessdata.strip() or None
    _ask_llama(cfg)
    groups = _ask_model_groups(cfg)
    save_local_settings(config_values(cfg))
    return cfg, groups


def _validate(cfg: AppConfig) -> None:
    report = run_diagnostics(cfg, visual=cfg.visual.enabled, report=True)
    print_readiness(report)


def run_setup_flow() -> int:
    banner("meeting-transcriber · подготовка")
    cfg = load_local_config()
    try:
        cfg, groups = _collect(cfg)
        _validate(cfg)
        while True:
            action = choose_from_options(
                "Что дальше?",
                [
                    Choice("Проверить снова", value="validate"),
                    Choice("Скачать модели", value="models"),
                    Choice("Установить llama.cpp", value="llama"),
                    Choice("Сохранить и выйти", value="save"),
                    Choice("Выход", value="exit"),
                ],
            )
            if action in {"save", "exit"} or action is None:
                save_local_settings(config_values(cfg))
                console.print("[green]Сохранено:[/green] config/local.yaml")
                return 0
            if action == "validate":
                _validate(cfg)
            elif action == "models":
                if not groups:
                    console.print("[yellow]Группы моделей не выбраны.[/yellow]")
                    continue
                manifest_path = cfg.resolve("models/manifest.yaml")
                if not manifest_path.is_file():
                    manifest_path = cfg.models_path("manifest.yaml")
                progress = DownloadProgress()
                try:
                    for group in groups:
                        download_models(
                            manifest_path,
                            cfg.models_path(),
                            only=group,
                            progress=progress,
                        )
                finally:
                    progress.close()
                _validate(cfg)
            elif action == "llama":
                try:
                    binary = install_llama(
                        cfg, lambda message: console.print(f"[muted]{message}[/muted]")
                    )
                    cfg.runtime.llama_server = str(binary)
                    save_local_settings(config_values(cfg))
                    console.print(f"[green]llama.cpp готов:[/green] {binary}")
                except LlamaError as exc:
                    console.print(f"[red]llama.cpp:[/red] {exc}")
                _validate(cfg)
    except KeyboardInterrupt:
        console.print("[muted]Отменено.[/muted]")
        return 130
    return 0


__all__ = ["run_setup_flow"]
