"""Rich-based CLI presentation: banners, stage progress and model selection."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

THEME = Theme(
    {
        "brand": "bold cyan",
        "muted": "dim",
        "ok": "bold green",
        "warn": "bold yellow",
        "error": "bold red",
        "stage": "bold white",
        "bar.complete": "green",
        "bar.finished": "green",
        "bar.pulse": "cyan",
        "prompt": "bold cyan",
        "prompt.choices": "cyan",
        "prompt.default": "dim cyan",
    }
)

console = Console(theme=THEME)
err_console = Console(theme=THEME, stderr=True)


def banner(title: str, subtitle: str | None = None) -> None:
    text = Text.from_markup(f"[brand]{title}[/brand]")
    if subtitle:
        text.append("\n")
        text.append_text(Text.from_markup(subtitle))
    console.print(Panel(text, border_style="cyan", padding=(0, 1)))


class StageProgress:
    """One row per pipeline stage, with spinner, bar and elapsed time."""

    def __init__(self, show: bool = True) -> None:
        self.show = show
        self._progress: Progress | None = None
        self._tasks: dict[str, Any] = {}
        self._labels: dict[str, str] = {}

    def __enter__(self) -> StageProgress:
        if self.show:
            self._progress = Progress(
                SpinnerColumn(style="cyan", finished_text="[green]✓[/green]"),
                TextColumn("[stage]{task.description}"),
                BarColumn(bar_width=None, pulse_style="bar.pulse"),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=err_console,
                transient=False,
            )
            self._progress.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._progress is not None:
            self._progress.stop()

    def _ensure(self, key: str, label: str) -> Any:
        self._labels[key] = label
        if not self.show or self._progress is None:
            return None
        if key not in self._tasks:
            self._tasks[key] = self._progress.add_task(
                f"[muted]{label}[/muted]", total=1, completed=0
            )
        return self._tasks[key]

    def start_stage(self, key: str, label: str) -> None:
        task = self._ensure(key, label)
        if task is not None and self._progress is not None:
            self._progress.update(
                task,
                description=f"[bold cyan]{label}[/bold cyan]",
                completed=0,
                total=1,
            )

    def finish_stage(self, key: str, label: str | None = None) -> None:
        task = self._ensure(key, label or self._labels.get(key, key))
        if task is not None and self._progress is not None:
            self._progress.update(
                task,
                description=f"[green]{label or self._labels.get(key, key)}[/green]",
                completed=1,
                total=1,
            )

    def skip_stage(self, key: str, label: str) -> None:
        task = self._ensure(key, label)
        if task is not None and self._progress is not None:
            self._progress.update(
                task,
                description=f"[muted]{label} (пропущено)[/muted]",
                completed=1,
                total=1,
            )

    def fail_stage(self, key: str, label: str | None = None) -> None:
        task = self._ensure(key, label or self._labels.get(key, key))
        if task is not None and self._progress is not None:
            self._progress.update(
                task,
                description=f"[red]{label or self._labels.get(key, key)}[/red]",
                completed=1,
                total=1,
            )


def result_panel(title: str, rows: list[tuple[str, str]]) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="muted", no_wrap=True)
    table.add_column(style="bold")
    for key, value in rows:
        table.add_row(key, value)
    console.print(Panel(table, title=f"[ok]{title}[/ok]", border_style="green"))


class DownloadProgress:
    """Progress adapter for ``models.download_models``."""

    def __init__(self) -> None:
        self._progress = Progress(
            SpinnerColumn(style="cyan", finished_text="[green]✓[/green]"),
            TextColumn("[stage]{task.description}"),
            BarColumn(bar_width=None, pulse_style="bar.pulse"),
            TaskProgressColumn(),
            TextColumn("[muted]{task.fields[amount]}[/muted]"),
            TimeElapsedColumn(),
            console=err_console,
        )
        self._tasks: dict[str, Any] = {}
        self._progress.start()

    def close(self) -> None:
        self._progress.stop()

    def group_start(self, key: str, name: str) -> None:
        err_console.print(f"[brand]◆[/brand] [bold]{key}[/bold] — {name}")

    def file_start(self, relative: str, size: int | None) -> None:
        self._tasks[relative] = self._progress.add_task(
            relative,
            total=size or None,
            amount=_human(size) if size else "",
        )

    def file_advance(self, relative: str, downloaded: int, total: int | None) -> None:
        task = self._tasks.get(relative)
        if task is None:
            return
        self._progress.update(
            task,
            completed=downloaded,
            total=total,
            amount=f"{_human(downloaded)} / {_human(total)}" if total else _human(downloaded),
        )

    def file_done(self, relative: str, ok: bool = True) -> None:
        task = self._tasks.get(relative)
        if task is None:
            return
        if ok:
            self._progress.update(
                task, description=f"[green]{relative}[/green]", completed=1, total=1
            )
        else:
            self._progress.update(
                task, description=f"[red]{relative}[/red]", completed=1, total=1
            )

    def file_skipped(self, relative: str) -> None:
        task = self._tasks.get(relative)
        if task is None:
            self._tasks[relative] = self._progress.add_task(
                f"[muted]{relative} (уже есть)[/muted]",
                total=1,
                completed=1,
                amount="",
            )
        else:
            self._progress.update(
                task,
                description=f"[muted]{relative} (уже есть)[/muted]",
                completed=1,
                total=1,
            )


def _human(size: float | None) -> str:
    if not size:
        return ""
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def choose_model_groups(groups: list[dict[str, Any]]) -> list[str]:
    """Show a table and ask which model groups to download."""

    table = Table(title="Модели meeting-transcriber", border_style="cyan")
    table.add_column("Ключ", style="bold cyan", no_wrap=True)
    table.add_column("Модель")
    table.add_column("Размер", justify="right", style="muted")
    table.add_column("Статус", justify="center")
    for group in groups:
        status = "[green]есть[/green]" if group["present"] else "[yellow]нет[/yellow]"
        table.add_row(group["key"], group["name"], group["size"], status)
    console.print(table)
    console.print(
        "[muted]Выберите группы для загрузки. Уже скачанные можно не отмечать.[/muted]"
    )
    selected: list[str] = []
    for group in groups:
        if Confirm.ask(
            f"Загрузить [bold]{group['key']}[/bold] — {group['name']}?",
            default=not group["present"],
            console=console,
        ):
            selected.append(group["key"])
    if selected and not Confirm.ask(
        f"Начать загрузку {len(selected)} групп(ы)?", default=True, console=console
    ):
        return []
    return selected


def ask_output_dir(default: str) -> str:
    return Prompt.ask(
        "Куда сохранить транскрипцию?",
        default=default,
        console=console,
        show_default=True,
    )


__all__ = [
    "DownloadProgress",
    "StageProgress",
    "ask_output_dir",
    "banner",
    "choose_model_groups",
    "console",
    "err_console",
    "result_panel",
]
