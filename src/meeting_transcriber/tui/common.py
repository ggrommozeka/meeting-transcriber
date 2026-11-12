"""Shared helpers for the questionary-based wizards."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import questionary
from questionary import Choice, Style

from ..setup.diagnostics import DiagnosticsReport
from ..ui import console

MEDIA_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".webm",
    ".mov",
    ".avi",
    ".m4v",
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".opus",
}

PROMPT_EXTENSIONS = {".md", ".txt", ".markdown"}

STYLE = Style(
    [
        ("qmark", "fg:#5fd7ff bold"),
        ("question", "bold"),
        ("answer", "fg:#8bd5ff"),
        ("pointer", "fg:#5fd7ff bold"),
        ("highlighted", "fg:#5fd7ff bold"),
        ("selected", "fg:#8bd5ff"),
        ("separator", "fg:#4b6478"),
        ("instruction", "fg:#7f8c99"),
        ("text", "fg:#d7dee8"),
        ("disabled", "fg:#5b6673 italic"),
    ]
)


def human_size(size: int | None) -> str:
    if not size:
        return "—"
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def media_filter(path: str) -> bool:
    return Path(path).suffix.lower() in MEDIA_EXTENSIONS


def prompt_filter(path: str) -> bool:
    return Path(path).suffix.lower() in PROMPT_EXTENSIONS


def discover_media_files(roots: list[Path], limit: int = 12) -> list[Path]:
    """Return existing media files from the given directories, newest first."""

    found: dict[Path, float] = {}
    for root in roots:
        root = root.expanduser()
        if root.is_file() and media_filter(str(root)):
            found[root.resolve()] = root.stat().st_mtime
            continue
        if not root.is_dir():
            continue
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_file() and media_filter(str(entry)):
                found[entry.resolve()] = entry.stat().st_mtime
    ordered = sorted(found.items(), key=lambda item: item[1], reverse=True)
    return [path for path, _ in ordered[:limit]]


def file_label(path: Path) -> str:
    try:
        stat = path.stat()
        size = human_size(stat.st_size)
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        size, modified = "—", "—"
    return f"{path.name}  [{size}, {modified}]  {path.parent}"


def validate_existing_file(value: str) -> bool | str:
    path = Path(value).expanduser()
    if not path.is_file():
        return f"Файл не найден: {path}"
    return True


def validate_existing_dir(value: str) -> bool | str:
    path = Path(value).expanduser()
    if not path.is_dir():
        return f"Папка не найдена: {path}"
    return True


def validate_media_file(value: str) -> bool | str:
    result = validate_existing_file(value)
    if result is not True:
        return result
    if not media_filter(value):
        return "Поддерживаются видео/аудио: mp4, mkv, webm, mov, wav, mp3, m4a, flac, ogg"
    return True


def validate_prompt_file(value: str) -> bool | str:
    result = validate_existing_file(value)
    if result is not True:
        return result
    if not prompt_filter(value):
        return "Промпт должен быть .md, .markdown или .txt"
    return True


def ask_path(
    message: str,
    default: str = "",
    *,
    directories_only: bool = False,
    media: bool = False,
    prompt: bool = False,
    style: Style = STYLE,
) -> Path | None:
    kwargs: dict[str, object] = {
        "default": default,
        "style": style,
        "only_directories": directories_only,
    }
    if directories_only:
        kwargs["validate"] = validate_existing_dir
    elif media:
        kwargs["validate"] = validate_media_file
        kwargs["file_filter"] = media_filter
    elif prompt:
        kwargs["validate"] = validate_prompt_file
        kwargs["file_filter"] = prompt_filter
    answer = questionary.path(message, **kwargs).ask()
    if answer is None:
        return None
    return Path(answer).expanduser().resolve()


def choose_from_options(
    message: str,
    options: list[Choice],
    *,
    style: Style = STYLE,
    shortcuts: bool = True,
) -> object | None:
    return questionary.select(
        message,
        choices=options,
        style=style,
        use_shortcuts=shortcuts,
        use_arrow_keys=True,
        use_jk_keys=True,
        instruction="↑↓ выбрать · Enter подтвердить · цифра для быстрого выбора",
    ).ask()


def print_readiness(report: DiagnosticsReport) -> None:
    from rich.table import Table

    table = Table(title="Готовность", border_style="cyan", show_lines=False)
    table.add_column("Компонент")
    table.add_column("Статус", justify="center")
    table.add_column("Готовность", justify="right")
    table.add_column("Детали", style="dim")
    for check in report.checks:
        style = {
            "ok": "green",
            "warn": "yellow",
            "fail": "red",
            "skipped": "dim",
        }.get(check.status, "white")
        table.add_row(
            check.label,
            f"[{style}]{check.status.upper()}[/{style}]",
            f"{check.progress:.0f}%",
            check.detail,
        )
    console.print(table)
    color = "red" if report.blocking else "green"
    console.print(
        f"[bold {color}]Итоговая готовность: {report.overall:.0f}%[/bold {color}]"
    )


__all__ = [
    "MEDIA_EXTENSIONS",
    "PROMPT_EXTENSIONS",
    "STYLE",
    "ask_path",
    "choose_from_options",
    "discover_media_files",
    "file_label",
    "human_size",
    "media_filter",
    "print_readiness",
    "prompt_filter",
    "validate_existing_dir",
    "validate_existing_file",
    "validate_media_file",
    "validate_prompt_file",
]
