"""Keyboard-first process wizard built with questionary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import questionary
from questionary import Choice

from ..config import AppConfig, ConfigError
from ..pipeline.runner import Pipeline, RunResult
from ..setup.config_store import load_local_config
from ..setup.diagnostics import run_diagnostics
from ..setup.llama_cpp import LlamaError, managed_llama_server
from ..ui import StageProgress, banner, console, result_panel
from .common import (
    STYLE,
    ask_path,
    choose_from_options,
    discover_media_files,
    file_label,
    print_readiness,
)

REPORT_LABELS = {
    "task_discussion": "Обсуждение задачи 1:1",
    "status_meeting": "Статус-митинг",
    "custom": "Свой промпт",
}


@dataclass
class ProcessSettings:
    input_path: Path | None = None
    output_dir: Path | None = None
    language: str = "ru"
    visual: bool = False
    diarization: bool = True
    report: bool = False
    keep_frames: bool = True
    speakers: int = 0
    report_mode: str = "general"
    custom_prompt: str | None = None


def _cancel() -> None:
    console.print("[muted]Отменено.[/muted]")


def ask_input(initial: Path | None = None) -> Path | None:
    roots = [Path.cwd(), Path("data/input").resolve()]
    if initial is not None:
        roots.insert(0, initial.parent)
    files = discover_media_files(roots)
    choices = [Choice(file_label(path), value=path) for path in files]
    choices += [
        Choice("Указать другой путь…", value="__path__"),
        Choice("Отмена", value="__cancel__"),
    ]
    selected = choose_from_options("Что обработать?", choices)
    if selected is None or selected == "__cancel__":
        return None
    if selected == "__path__":
        return ask_path(
            "Путь к видео или аудио",
            default=str(initial or Path.cwd()),
            media=True,
        )
    return selected  # type: ignore[return-value]


def ask_output(settings: ProcessSettings, cfg: AppConfig) -> Path | None:
    assert settings.input_path is not None
    default_dir = settings.input_path.parent
    project_output = cfg.resolve(cfg.paths.output_dir)
    choices = [
        Choice(f"Рядом с файлом — {default_dir}", value=default_dir),
    ]
    if project_output != default_dir:
        choices.append(Choice(f"В папку проекта — {project_output}", value=project_output))
    choices += [
        Choice("Указать другую папку…", value="__path__"),
        Choice("Отмена", value="__cancel__"),
    ]
    selected = choose_from_options("Куда сохранить результат?", choices)
    if selected is None or selected == "__cancel__":
        return None
    if selected == "__path__":
        return ask_path(
            "Выходная папка", default=str(default_dir), directories_only=True
        )
    return selected  # type: ignore[return-value]


def ask_features(settings: ProcessSettings) -> bool:
    choices = [
        Choice("OCR видео (слайды и код)", value="visual", checked=settings.visual),
        Choice(
            "Диаризация (спикеры)",
            value="diarization",
            checked=settings.diarization,
        ),
        Choice("LLM-отчёт", value="report", checked=settings.report),
        Choice(
            "Сохранять кадры кода",
            value="keep-frames",
            checked=settings.keep_frames,
        ),
    ]
    selected = questionary.checkbox(
        "Что включить?",
        choices=choices,
        style=STYLE,
        instruction="Space — переключить · Enter — далее",
    ).ask()
    if selected is None:
        return False
    settings.visual = "visual" in selected
    settings.diarization = "diarization" in selected
    settings.report = "report" in selected
    settings.keep_frames = "keep-frames" in selected
    return True


def ask_speakers(settings: ProcessSettings) -> bool:
    if not settings.diarization:
        settings.speakers = 0
        return True
    value = questionary.text(
        "Сколько спикеров? 0 = авто",
        default=str(settings.speakers or 0),
        style=STYLE,
        validate=lambda text: (
            True if text.strip().isdigit() and int(text) >= 0 else "Введите число >= 0"
        ),
    ).ask()
    if value is None:
        return False
    settings.speakers = int(value.strip())
    return True


def ask_report(settings: ProcessSettings) -> bool:
    if not settings.report:
        settings.report_mode = "general"
        settings.custom_prompt = None
        return True
    choices = [
        Choice(
            "Обсуждение задачи 1:1",
            value="task_discussion",
            description="Задачи, C++/архитектура, критерии приёмки",
        ),
        Choice(
            "Статус-митинг",
            value="status_meeting",
            description="Прогресс по задачам, блокеры, новые задачи",
        ),
        Choice(
            "Свой промпт…",
            value="custom",
            description="Свободные инструкции на один запуск",
        ),
        Choice("Без отчёта", value="none"),
    ]
    selected = choose_from_options("Режим отчёта", choices)
    if selected is None:
        return False
    if selected == "none":
        settings.report = False
        settings.report_mode = "general"
        settings.custom_prompt = None
        return True
    settings.report_mode = str(selected)
    if settings.report_mode == "custom":
        return ask_custom_prompt(settings)
    settings.custom_prompt = None
    return True


def ask_custom_prompt(settings: ProcessSettings) -> bool:
    method = choose_from_options(
        "Как задать промпт?",
        [
            Choice("Ввести текст…", value="text"),
            Choice("Загрузить из файла…", value="file"),
            Choice("Назад", value="back"),
        ],
    )
    if method is None or method == "back":
        return False
    if method == "file":
        path = ask_path("Файл с промптом", prompt=True)
        if path is None:
            return False
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            console.print(f"[red]Не удалось прочитать файл:[/red] {exc}")
            return False
    else:
        text = questionary.text(
            "Введите промпт",
            multiline=True,
            style=STYLE,
            instruction="Esc+Enter — применить",
        ).ask()
        if text is None:
            return False
    text = text.strip()
    if not text:
        console.print("[red]Промпт пустой.[/red]")
        return False
    limit = load_local_config().llm.prompt_max_chars
    if len(text) > limit:
        console.print(f"[red]Промпт больше лимита {limit} символов.[/red]")
        return False
    settings.custom_prompt = text
    return True


def apply_settings(cfg: AppConfig, settings: ProcessSettings) -> AppConfig:
    assert settings.input_path is not None
    assert settings.output_dir is not None
    cfg.language = settings.language
    cfg.visual.enabled = settings.visual
    cfg.visual.keep_code_frames = settings.keep_frames
    cfg.diarization.enabled = settings.diarization
    cfg.diarization.num_speakers = settings.speakers or None
    cfg.llm.enabled = settings.report
    cfg.llm.report_mode = settings.report_mode
    cfg.llm.custom_prompt = settings.custom_prompt
    cfg.paths.output_dir = str(settings.output_dir)
    return cfg


def summary_rows(settings: ProcessSettings) -> list[tuple[str, str]]:
    assert settings.input_path is not None
    assert settings.output_dir is not None
    features = []
    if settings.visual:
        features.append("OCR")
    if settings.diarization:
        features.append("диаризация")
    if settings.report:
        features.append(REPORT_LABELS.get(settings.report_mode, settings.report_mode))
    return [
        ("Вход", str(settings.input_path)),
        ("Выход", str(settings.output_dir)),
        ("Язык", settings.language),
        ("Спикеры", "авто" if not settings.speakers else str(settings.speakers)),
        ("Функции", ", ".join(features) or "только транскрипция"),
    ]


def print_summary(settings: ProcessSettings) -> None:
    from rich.panel import Panel
    from rich.table import Table

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="muted", no_wrap=True)
    table.add_column(style="bold")
    for key, value in summary_rows(settings):
        table.add_row(key, value)
    console.print(Panel(table, title="[ok]Параметры запуска[/ok]", border_style="cyan"))


def confirm_and_run(cfg: AppConfig, settings: ProcessSettings) -> int:
    assert settings.input_path is not None
    assert settings.output_dir is not None
    report_obj = run_diagnostics(
        cfg,
        visual=settings.visual,
        diarization=settings.diarization,
        report=settings.report,
        llama_server=cfg.runtime.llama_server,
    )
    print_readiness(report_obj)
    if report_obj.blocking:
        console.print("[red]Есть блокирующие проблемы.[/red]")
        action = choose_from_options(
            "Что сделать?",
            [
                Choice("Изменить настройки", value="edit"),
                Choice("Отмена", value="cancel"),
            ],
        )
        return 0 if action == "cancel" else -1

    print_summary(settings)
    action = choose_from_options(
        "Продолжить?",
        [
            Choice("Запустить", value="run"),
            Choice("Изменить настройки", value="edit"),
            Choice("Отмена", value="cancel"),
        ],
    )
    if action != "run":
        return 0 if action == "cancel" else -1

    def report_server():
        return managed_llama_server(
            cfg,
            progress=lambda message: console.print(f"[muted]{message}[/muted]"),
            log_path=cfg.resolve("logs/llama-server-questionary.log"),
            pid_path=cfg.resolve("logs/llama-server.pid"),
        )

    try:
        with StageProgress(show=True) as progress:
            result: RunResult = Pipeline(
                cfg,
                skip_llm=not settings.report,
                progress=progress,
                report_server=report_server if settings.report else None,
            ).run(settings.input_path)
        console.print()
        result_panel(
            "Готово",
            [
                ("Вход", str(settings.input_path)),
                ("Выход", str(settings.output_dir)),
                ("Сегменты", str(len(result.transcript.segments))),
                ("Спикеры", str(len(result.transcript.speaker_ids))),
            ],
        )
        for name, path in result.outputs.items():
            console.print(f"  [brand]{name:<14}[/brand] [muted]{path}[/muted]")
        return 0
    except (LlamaError, ConfigError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 1


def run_process_flow(
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> int:
    banner("meeting-transcriber · обработка")
    cfg = load_local_config()
    settings = ProcessSettings(
        input_path=Path(input_path).expanduser().resolve() if input_path else None,
        output_dir=Path(output_dir).expanduser().resolve() if output_dir else None,
    )
    while True:
        if settings.input_path is None:
            settings.input_path = ask_input()
            if settings.input_path is None:
                _cancel()
                return 0
        if settings.output_dir is None:
            settings.output_dir = ask_output(settings, cfg)
            if settings.output_dir is None:
                _cancel()
                return 0
        if not ask_features(settings):
            _cancel()
            return 0
        if not ask_speakers(settings):
            _cancel()
            return 0
        if not ask_report(settings):
            _cancel()
            return 0
        cfg = apply_settings(cfg, settings)
        code = confirm_and_run(cfg, settings)
        if code >= 0:
            return code
        # "Изменить настройки" — повторить с сохранёнными значениями.


__all__ = ["ProcessSettings", "run_process_flow"]
