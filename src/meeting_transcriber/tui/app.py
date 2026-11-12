"""Textual TUI: setup wizard and processing wizard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    ContentSwitcher,
    DataTable,
    DirectoryTree,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RadioButton,
    RadioSet,
    RichLog,
    Select,
    Static,
    Switch,
    TextArea,
)

from ..config import AppConfig
from ..models import download_models
from ..pipeline.runner import Pipeline, RunResult
from ..setup.config_store import config_values, load_local_config, save_local_settings
from ..setup.diagnostics import DiagnosticsReport, run_diagnostics
from ..setup.llama_cpp import (
    LlamaError,
    install_llama,
    is_server_running,
    resolve_llama_binary,
    start_server,
    stop_server,
)

SETUP_CSS = """
Screen { background: #10151c; color: #d7dee8; }
Header { background: #16202b; color: #9fd6ff; }
Footer { background: #16202b; color: #9aa7b4; }
.panel { border: round #2a3a4a; padding: 1 2; margin: 1; background: #131b24; }
.title { color: #8bd5ff; text-style: bold; margin-bottom: 1; }
.row { height: auto; }
.field { width: 1fr; margin-right: 2; }
.wide { width: 100%; }
Button { margin-right: 1; }
#checks { height: 14; border: round #2a3a4a; }
#log { height: 10; border: round #2a3a4a; background: #0d1319; }
ProgressBar { margin: 1 0; }
.muted { color: #8b9bab; }
#report-modal { align: center middle; }
#report-dialog { width: 96; height: 38; border: round #3a5a72; background: #131b24; padding: 1 2; }
#report-dialog .title { margin-bottom: 1; }
#report-mode { height: auto; }
#report-switcher { height: 12; margin: 1 0; }
#prompt-editor { height: 10; border: round #2a3a4a; }
#prompt-counter { height: 1; color: #8b9bab; }
#file-dialog { width: 80; height: 30; border: round #3a5a72; background: #131b24; padding: 1 2; }
#file-tree { height: 20; }
"""


def _status_text(status: str) -> Text:
    return {
        "ok": Text("OK", style="bold green"),
        "warn": Text("WARN", style="bold yellow"),
        "fail": Text("FAIL", style="bold red"),
        "skipped": Text("—", style="dim"),
    }.get(status, Text(status))


def _render_table(table: DataTable, report: DiagnosticsReport) -> None:
    table.clear()
    for check in report.checks:
        table.add_row(
            check.label,
            _status_text(check.status),
            f"{check.progress:.0f}%",
            check.detail,
        )


class _TuiProgress:
    """Pipeline progress adapter that marshals updates to the Textual app."""

    _ORDER = [
        ("preprocess", "Извлечение аудио"),
        ("visual", "Анализ кадров"),
        ("asr", "Распознавание речи"),
        ("diarization", "Диаризация"),
        ("alignment", "Выравнивание"),
        ("terminology", "Терминология"),
        ("llm_report", "LLM-отчёт"),
        ("outputs", "Сохранение"),
    ]

    def __init__(self, app: ProcessApp) -> None:
        self.app = app
        self.done: set[str] = set()
        self.total = len(self._ORDER)

    def _event(self, key: str, label: str, event: str) -> None:
        if event == "done":
            self.done.add(key)
        self.app.call_from_thread(self.app.stage_event, key, label, event, len(self.done), self.total)

    def start_stage(self, key: str, label: str) -> None:
        self._event(key, label, "start")

    def finish_stage(self, key: str, label: str | None = None) -> None:
        self._event(key, label or key, "done")

    def skip_stage(self, key: str, label: str) -> None:
        self._event(key, label, "skip")

    def fail_stage(self, key: str, label: str | None = None) -> None:
        self._event(key, label or key, "fail")


class _TuiDownloadProgress:
    def __init__(self, app: SetupApp) -> None:
        self.app = app
        self._last: dict[str, int] = {}

    def _write(self, message: str) -> None:
        self.app.call_from_thread(self.app._log, message)

    def group_start(self, key: str, name: str) -> None:
        self._write(f"[bold cyan]{key}[/bold cyan] — {name}")

    def file_start(self, relative: str, size: int | None) -> None:
        self._write(f"→ {relative}")

    def file_advance(self, relative: str, downloaded: int, total: int | None) -> None:
        if not total:
            return
        percent = int(downloaded * 100 / total)
        last = self._last.get(relative, -10)
        if percent >= last + 10:
            self._last[relative] = percent
            self._write(f"  {relative}: {percent}%")

    def file_done(self, relative: str, ok: bool = True) -> None:
        self._write(("[green]✓[/green] " if ok else "[red]✗[/red] ") + relative)

    def file_skipped(self, relative: str) -> None:
        self._write(f"[dim]✓ {relative} (уже есть)[/dim]")


class BaseWizard(App):
    CSS = SETUP_CSS
    BINDINGS = [("q", "quit", "Выход")]

    def _log(self, message: str) -> None:
        self.query_one("#log", RichLog).write(message)

    def _report_table(self, report: DiagnosticsReport) -> None:
        _render_table(self.query_one("#checks", DataTable), report)
        self.query_one("#overall", Static).update(
            f"[bold]Общая готовность:[/bold] [bold green]{report.overall:.0f}%[/bold green]"
            + (
                f"  [red]Блокирующих проблем: {len(report.blocking)}[/red]"
                if report.blocking
                else "  [green]всё готово[/green]"
            )
        )


class SetupApp(BaseWizard):
    TITLE = "meeting-transcriber · подготовка"

    def compose(self) -> ComposeResult:
        yield Header()
        with ScrollableContainer():
            with Vertical(classes="panel"):
                yield Label("Подготовка окружения", classes="title")
                with Horizontal(classes="row"):
                    yield Label("Каталог моделей", classes="field")
                    yield Input(id="models-dir", classes="wide")
                with Horizontal(classes="row"):
                    yield Label("OCR видео", classes="field")
                    yield Switch(id="visual-enabled")
                with Horizontal(classes="row"):
                    yield Label("Tesseract", classes="field")
                    yield Input(placeholder="tesseract", id="tesseract", classes="wide")
                with Horizontal(classes="row"):
                    yield Label("Tessdata", classes="field")
                    yield Input(placeholder="пусто = системный путь", id="tessdata", classes="wide")

            with Vertical(classes="panel"):
                yield Label("llama.cpp", classes="title")
                with Horizontal(classes="row"):
                    yield Label("Источник", classes="field")
                    yield Select(
                        [
                            ("Готовый бинарник", "existing"),
                            ("Скачать релиз", "release"),
                            ("Собрать из исходников", "build"),
                        ],
                        value="existing",
                        id="llama-source",
                        classes="wide",
                    )
                with Horizontal(classes="row"):
                    yield Label("Путь к llama-server", classes="field")
                    yield Input(placeholder="/path/to/llama-server", id="llama-path", classes="wide")
                with Horizontal(classes="row"):
                    yield Label("Тег", classes="field")
                    yield Input(value="latest", id="llama-tag", classes="wide")
                with Horizontal(classes="row"):
                    yield Label("Каталог сборки", classes="field")
                    yield Input(id="llama-build-dir", classes="wide")
                with Horizontal(classes="row"):
                    yield Label("CUDA", classes="field")
                    yield Switch(value=True, id="llama-cuda")
                    yield Label("Jobs", classes="field")
                    yield Input(value="0", id="llama-jobs", classes="wide")
                with Horizontal(classes="row"):
                    yield Label("Группы моделей", classes="field")
                    yield Switch(value=True, id="model-asr")
                    yield Label("ASR")
                    yield Switch(value=True, id="model-diarization")
                    yield Label("Diarization")
                    yield Switch(value=False, id="model-llm")
                    yield Label("LLM")

            with Horizontal(classes="row"):
                yield Button("Проверить", id="validate", variant="primary")
                yield Button("Сохранить настройки", id="save")
                yield Button("Скачать модели", id="download")
                yield Button("Установить llama.cpp", id="llama-install")
            yield Static(id="overall")
            yield DataTable(id="checks")
            yield RichLog(id="log", markup=True)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#checks", DataTable)
        table.add_columns("Компонент", "Статус", "Готовность", "Детали")
        self._load_config()
        self._validate_worker()

    def _load_config(self) -> None:
        cfg = load_local_config()
        self.query_one("#models-dir", Input).value = cfg.paths.models_dir
        self.query_one("#visual-enabled", Switch).value = cfg.visual.enabled
        self.query_one("#tesseract", Input).value = cfg.visual.tesseract
        self.query_one("#tessdata", Input).value = cfg.visual.tessdata_dir or ""
        self.query_one("#llama-source", Select).value = cfg.runtime.llama_source
        self.query_one("#llama-path", Input).value = cfg.runtime.llama_server or ""
        self.query_one("#llama-tag", Input).value = cfg.runtime.llama_tag
        self.query_one("#llama-build-dir", Input).value = cfg.runtime.llama_build_dir
        self.query_one("#llama-cuda", Switch).value = cfg.runtime.llama_cuda
        self.query_one("#llama-jobs", Input).value = str(cfg.runtime.llama_jobs)

    def _sync(self) -> AppConfig:
        cfg = load_local_config()
        cfg.paths.models_dir = self.query_one("#models-dir", Input).value.strip() or "models"
        cfg.visual.enabled = self.query_one("#visual-enabled", Switch).value
        cfg.visual.tesseract = self.query_one("#tesseract", Input).value.strip() or "tesseract"
        tessdata = self.query_one("#tessdata", Input).value.strip()
        cfg.visual.tessdata_dir = tessdata or None
        cfg.runtime.llama_source = str(self.query_one("#llama-source", Select).value)
        cfg.runtime.llama_server = self.query_one("#llama-path", Input).value.strip() or None
        cfg.runtime.llama_tag = self.query_one("#llama-tag", Input).value.strip() or "latest"
        cfg.runtime.llama_build_dir = (
            self.query_one("#llama-build-dir", Input).value.strip()
            or "~/.local/llama.cpp"
        )
        cfg.runtime.llama_cuda = self.query_one("#llama-cuda", Switch).value
        try:
            cfg.runtime.llama_jobs = int(self.query_one("#llama-jobs", Input).value or 0)
        except ValueError:
            cfg.runtime.llama_jobs = 0
        return cfg

    @work(thread=True, exclusive=True)
    def _validate_worker(self) -> None:
        try:
            cfg = self._sync()
            report = run_diagnostics(cfg, visual=cfg.visual.enabled, report=True)
            self.call_from_thread(self._report_table, report)
            self.call_from_thread(self._log, f"Проверка завершена: {report.overall:.0f}%")
        except Exception as exc:
            self.call_from_thread(self._log, f"[red]Ошибка проверки:[/red] {exc}")

    @work(thread=True, exclusive=True)
    def _llama_worker(self) -> None:
        cfg = self._sync()
        try:
            binary = install_llama(cfg, lambda message: self.call_from_thread(self._log, message))
            self.call_from_thread(self._set_llama_path, str(binary))
            self.call_from_thread(self._log, f"[green]llama.cpp готов:[/green] {binary}")
            self.call_from_thread(self._save_config)
        except LlamaError as exc:
            self.call_from_thread(self._log, f"[red]llama.cpp:[/red] {exc}")
        except Exception as exc:
            self.call_from_thread(self._log, f"[red]llama.cpp:[/red] {exc}")

    def _set_llama_path(self, value: str) -> None:
        self.query_one("#llama-path", Input).value = value

    def _save_config(self) -> None:
        cfg = self._sync()
        path = save_local_settings(config_values(cfg))
        self._log(f"[green]Сохранено:[/green] {path}")

    @work(thread=True, exclusive=True)
    def _download_worker(self) -> None:
        cfg = self._sync()
        manifest_path = cfg.resolve("models/manifest.yaml")
        if not manifest_path.is_file():
            manifest_path = cfg.models_path("manifest.yaml")
        groups = []
        for key, widget in (
            ("asr", "#model-asr"),
            ("diarization", "#model-diarization"),
            ("llm", "#model-llm"),
        ):
            if self.query_one(widget, Switch).value:
                groups.append(key)
        if not groups:
            self.call_from_thread(self._log, "[yellow]Ни одна группа моделей не выбрана[/yellow]")
            return
        try:
            for key in groups:
                code = download_models(
                    manifest_path,
                    cfg.models_path(),
                    only=key,
                    progress=_TuiDownloadProgress(self),
                )
                self.call_from_thread(self._log, f"{key}: {'OK' if code == 0 else 'ошибка'}")
        except Exception as exc:
            self.call_from_thread(self._log, f"[red]Загрузка моделей:[/red] {exc}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "validate":
            self._validate_worker()
        elif event.button.id == "save":
            self._save_config()
        elif event.button.id == "download":
            self._download_worker()
        elif event.button.id == "llama-install":
            self._llama_worker()


@dataclass
class ReportChoice:
    mode: str
    custom_prompt: str | None = None


class PromptFilePicker(ModalScreen[Path | None]):
    """Small file picker for .md/.txt prompt files."""

    BINDINGS = [("escape", "cancel", "Отмена")]

    def compose(self) -> ComposeResult:
        with Vertical(id="file-dialog"):
            yield Label("Выберите файл с промптом", classes="title")
            yield DirectoryTree(Path.cwd(), id="file-tree")
            yield Button("Отмена", id="file-cancel")

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        self.dismiss(Path(event.path))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "file-cancel":
            self.dismiss(None)


class ReportOptionsScreen(ModalScreen[ReportChoice]):
    """Choose report mode or enter a one-off custom prompt."""

    BINDINGS = [("escape", "cancel", "Назад")]
    _MODES = ["task_discussion", "status_meeting", "custom"]
    _HELP = {
        "task_discussion": (
            "[bold]Обсуждение задачи 1:1[/bold]\n"
            "Черновик задачи: цель, границы, требования, C++-модули, API/ABI, "
            "критерии приёмки и открытые вопросы. Речевой шум отфильтровывается.\n\n"
            "[yellow]Рекомендуется включить OCR видео[/yellow], чтобы получить "
            "точные имена классов и файлов."
        ),
        "status_meeting": (
            "[bold]Статус-митинг[/bold]\n"
            "Прогресс по задачам: кто, что сделал, что дальше, какие блокеры. "
            "Отдельно фиксируются новые задачи.\n\n"
            "[yellow]Для имён вместо speaker_id заполните speakers.yaml[/yellow]."
        ),
        "custom": (
            "[bold]Свой промпт[/bold]\n"
            "Произвольные инструкции добавляются к системному промпту. "
            "JSON-схема, ссылки S<n> и C<n> сохраняются."
        ),
    }

    def __init__(self, initial: ReportChoice | None = None) -> None:
        super().__init__()
        self.initial = initial
        self.max_chars = load_local_config().llm.prompt_max_chars

    def compose(self) -> ComposeResult:
        with Vertical(id="report-dialog"):
            yield Label("Параметры отчёта", classes="title")
            yield RadioSet(
                RadioButton("Обсуждение задачи 1:1", id="mode-task", value=True),
                RadioButton("Статус-митинг", id="mode-status"),
                RadioButton("Свой промпт", id="mode-custom"),
                id="report-mode",
            )
            yield ContentSwitcher(
                Static(self._HELP["task_discussion"], id="mode-help"),
                Vertical(
                    TextArea(
                        id="prompt-editor",
                        placeholder="Опишите фокус, структуру и стиль отчёта…",
                        soft_wrap=True,
                    ),
                    Horizontal(
                        Button("Выбрать файл", id="prompt-file"),
                        Static("0 / 6000", id="prompt-counter"),
                    ),
                    id="custom-panel",
                ),
                initial="mode-help",
                id="report-switcher",
            )
            with Horizontal(classes="row"):
                yield Button("Назад", id="report-back")
                yield Button("Продолжить", id="report-continue", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        if self.initial and self.initial.mode == "custom":
            self.query_one("#mode-custom", RadioButton).value = True
            self.query_one("#report-switcher", ContentSwitcher).current = "custom-panel"
            editor = self.query_one("#prompt-editor", TextArea)
            editor.load_text(self.initial.custom_prompt or "")
            self._update_counter()
        elif self.initial and self.initial.mode in self._MODES:
            button_id = {
                "task_discussion": "#mode-task",
                "status_meeting": "#mode-status",
            }.get(self.initial.mode)
            if button_id:
                self.query_one(button_id, RadioButton).value = True

    def _update_counter(self) -> None:
        length = len(self.query_one("#prompt-editor", TextArea).text)
        self.query_one("#prompt-counter", Static).update(f"{length} / {self.max_chars}")
        color = "red" if length > self.max_chars else "#8b9bab"
        self.query_one("#prompt-counter", Static).styles.color = color

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        index = event.index
        if index < 0 or index >= len(self._MODES):
            return
        mode = self._MODES[index]
        switcher = self.query_one("#report-switcher", ContentSwitcher)
        switcher.current = "custom-panel" if mode == "custom" else "mode-help"
        if mode != "custom":
            self.query_one("#mode-help", Static).update(self._HELP[mode])

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id == "prompt-editor":
            self._update_counter()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "report-back":
            self.dismiss(None)
        elif event.button.id == "prompt-file":
            self.app.push_screen(PromptFilePicker(), self._file_picked)
        elif event.button.id == "report-continue":
            self._continue()

    def _file_picked(self, path: Path | None) -> None:
        if path is None:
            return
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            self.notify(f"Не удалось прочитать файл: {exc}", severity="error")
            return
        if len(text) > self.max_chars:
            self.notify(
                f"Файл больше лимита: {len(text)} > {self.max_chars}",
                severity="error",
            )
            return
        self.query_one("#prompt-editor", TextArea).load_text(text)
        self._update_counter()

    def _continue(self) -> None:
        index = self.query_one("#report-mode", RadioSet).pressed_index
        if index < 0 or index >= len(self._MODES):
            return
        mode = self._MODES[index]
        prompt = self.query_one("#prompt-editor", TextArea).text.strip()
        if mode == "custom":
            if not prompt:
                self.notify("Введите текст промпта или выберите файл", severity="error")
                return
            if len(prompt) > self.max_chars:
                self.notify("Промпт превышает лимит", severity="error")
                return
        self.dismiss(
            ReportChoice(mode=mode, custom_prompt=prompt if mode == "custom" else None)
        )

    def action_cancel(self) -> None:
        self.dismiss(None)


class ProcessApp(BaseWizard):
    TITLE = "meeting-transcriber · обработка"

    def __init__(self, input_path: str | Path | None = None, output_dir: str | Path | None = None) -> None:
        super().__init__()
        self._initial_input = str(input_path) if input_path else ""
        self._initial_output = str(output_dir) if output_dir else ""
        self._report_choice: ReportChoice | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with ScrollableContainer():
            with Vertical(classes="panel"):
                yield Label("Вход и выход", classes="title")
                with Horizontal(classes="row"):
                    yield Label("Входной файл", classes="field")
                    yield Input(placeholder="/path/to/meeting.mp4", id="input", classes="wide")
                    yield Button("Выбрать…", id="input-pick")
                with Horizontal(classes="row"):
                    yield Label("Выходная папка", classes="field")
                    yield Input(placeholder="по умолчанию рядом с файлом", id="output", classes="wide")
                    yield Button("Выбрать…", id="output-pick")
                with Horizontal(classes="row"):
                    yield Label("Язык", classes="field")
                    yield Input(value="ru", id="language", classes="wide")
                with Horizontal(classes="row"):
                    yield Label("Спикеров (0 = авто)", classes="field")
                    yield Input(value="0", id="speakers", classes="wide")

            with Vertical(classes="panel"):
                yield Label("Что делать", classes="title")
                with Horizontal(classes="row"):
                    yield Label("OCR видео", classes="field")
                    yield Switch(id="visual")
                    yield Label("Диаризация")
                    yield Switch(value=True, id="diarization")
                    yield Label("LLM-отчёт")
                    yield Switch(id="report")
                yield Static(
                    "Режим отчёта выбирается перед запуском",
                    id="report-summary",
                    classes="muted",
                )
                with Horizontal(classes="row"):
                    yield Label("Стиль отчёта", classes="field")
                    yield Select(
                        [("Кратко (ASD-STE100)", "asd-ste100"), ("Полный", "plain")],
                        value="asd-ste100",
                        id="style",
                        classes="wide",
                    )
                    yield Label("Кадры кода")
                    yield Switch(value=True, id="keep-frames")
                with Horizontal(classes="row"):
                    yield Label("Профиль терминов", classes="field")
                    yield Input(value="general-it", id="profile", classes="wide")

            with Horizontal(classes="row"):
                yield Button("Проверить", id="validate", variant="primary")
                yield Button("Запустить", id="run", variant="success")
            yield Static(id="overall")
            yield ProgressBar(total=100, show_eta=False, id="bar")
            yield Static(id="stage", classes="muted")
            yield DataTable(id="checks")
            yield RichLog(id="log", markup=True)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#checks", DataTable)
        table.add_columns("Компонент", "Статус", "Готовность", "Детали")
        if self._initial_input:
            self.query_one("#input", Input).value = self._initial_input
        if self._initial_output:
            self.query_one("#output", Input).value = self._initial_output
        self._validate_worker()

    def _sync(self) -> AppConfig:
        cfg = load_local_config()
        cfg.language = self.query_one("#language", Input).value.strip() or "ru"
        cfg.visual.enabled = self.query_one("#visual", Switch).value
        cfg.diarization.enabled = self.query_one("#diarization", Switch).value
        cfg.llm.enabled = self.query_one("#report", Switch).value
        if self._report_choice is not None:
            cfg.llm.report_mode = self._report_choice.mode
            cfg.llm.custom_prompt = self._report_choice.custom_prompt
        cfg.llm.report_style = str(self.query_one("#style", Select).value)
        cfg.visual.keep_code_frames = self.query_one("#keep-frames", Switch).value
        cfg.terminology.profiles = [
            self.query_one("#profile", Input).value.strip() or "general-it"
        ]
        try:
            speakers = int(self.query_one("#speakers", Input).value or 0)
        except ValueError:
            speakers = 0
        cfg.diarization.num_speakers = speakers or None
        return cfg

    def _paths(self) -> tuple[Path | None, Path | None]:
        raw = self.query_one("#input", Input).value.strip()
        if not raw:
            return None, None
        source = Path(raw).expanduser().resolve()
        out_raw = self.query_one("#output", Input).value.strip()
        output = Path(out_raw).expanduser().resolve() if out_raw else source.parent
        return source, output

    @work(thread=True, exclusive=True)
    def _validate_worker(self) -> None:
        try:
            cfg = self._sync()
            source, output = self._paths()
            report = run_diagnostics(
                cfg,
                visual=cfg.visual.enabled,
                diarization=cfg.diarization.enabled,
                report=cfg.llm.enabled,
                llama_server=cfg.runtime.llama_server,
            )
            extra = []
            if source is None:
                extra.append(("Входной файл", "warn", 0, "нажмите «Выбрать…»"))
            elif not source.is_file():
                extra.append(("Входной файл", "fail", 0, f"не найден: {source}"))
            else:
                extra.append(("Входной файл", "ok", 100, str(source)))
            if output is not None:
                try:
                    output.mkdir(parents=True, exist_ok=True)
                    extra.append(("Выходная папка", "ok", 100, str(output)))
                except OSError as exc:
                    extra.append(("Выходная папка", "fail", 0, str(exc)))
            self.call_from_thread(self._report_with_extra, report, extra)
        except Exception as exc:
            self.call_from_thread(self._log, f"[red]Ошибка проверки:[/red] {exc}")

    def _report_with_extra(self, report: DiagnosticsReport, extra: list[tuple]) -> None:
        from ..setup.diagnostics import Check

        for label, status, progress, detail in extra:
            report.checks.append(Check(label, label, status, progress, detail, 10))
        self._report_table(report)
        self._log(f"Проверка завершена: {report.overall:.0f}%")

    def stage_event(
        self, key: str, label: str, event: str, done: int, total: int
    ) -> None:
        self.query_one("#bar", ProgressBar).progress = done * 100 / total
        prefix = {"start": "→", "done": "✓", "skip": "—", "fail": "✗"}.get(event, "·")
        self.query_one("#stage", Static).update(f"{prefix} {label}")
        self._log(f"{prefix} {label}")

    @work(thread=True, exclusive=True)
    def _run_worker(self) -> None:
        cfg = self._sync()
        source, output = self._paths()
        if source is None or not source.is_file():
            self.call_from_thread(self._log, "[red]Укажите существующий входной файл[/red]")
            return
        assert output is not None
        cfg.paths.output_dir = str(output)
        server = None
        try:
            if cfg.llm.enabled and not is_server_running(cfg.llm.base_url):
                binary = resolve_llama_binary(cfg)
                if binary is None:
                    raise LlamaError("LLM-отчёт включён, но llama-server не найден")
                model = cfg.models_path(
                    "qwen3.5-9b-gguf", "Qwen3.5-9B-UD-Q4_K_XL.gguf"
                )
                server = start_server(
                    binary,
                    model,
                    base_url=cfg.llm.base_url,
                    ctx=cfg.llm.num_ctx,
                    log_path=cfg.resolve("logs/llama-server-tui.log"),
                    progress=lambda message: self.call_from_thread(self._log, message),
                )
            app = self
            progress = _TuiProgress(app)
            result = Pipeline(
                cfg,
                skip_llm=not cfg.llm.enabled,
                progress=progress,
            ).run(source)
            self.call_from_thread(self._finish, result)
        except Exception as exc:
            self.call_from_thread(self._log, f"[red]Ошибка:[/red] {exc}")
        finally:
            if server is not None:
                stop_server(server)
                self.call_from_thread(self._log, "llama-server остановлен")

    def _finish(self, result: RunResult) -> None:
        self.query_one("#bar", ProgressBar).progress = 100
        self.query_one("#stage", Static).update("готово")
        self._log("[bold green]Обработка завершена[/bold green]")
        for name, path in result.outputs.items():
            self._log(f"{name}: {path}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "input-pick":
            self.push_screen(
                FilePicker(
                    start=self.query_one("#input", Input).value or Path.cwd(),
                    mode="file",
                    title="Выберите видео или аудио",
                ),
                self._input_picked,
            )
        elif event.button.id == "output-pick":
            self.push_screen(
                FilePicker(
                    start=self.query_one("#output", Input).value or Path.cwd(),
                    mode="directory",
                    title="Выберите выходную папку",
                ),
                self._output_picked,
            )
        elif event.button.id == "validate":
            self._validate_worker()
        elif event.button.id == "run":
            if self.query_one("#report", Switch).value:
                self.push_screen(ReportOptionsScreen(self._report_choice), self._report_chosen)
            else:
                self._report_choice = None
                self._run_worker()

    def _report_chosen(self, choice: ReportChoice | None) -> None:
        if choice is None:
            return
        self._report_choice = choice
        label = {
            "task_discussion": "Обсуждение задачи 1:1",
            "status_meeting": "Статус-митинг",
            "custom": "Свой промпт",
        }.get(choice.mode, choice.mode)
        self.query_one("#report-summary", Static).update(f"Режим отчёта: {label}")
        self._log(f"Режим отчёта: {label}")
        self._run_worker()


def run_setup() -> None:
    SetupApp().run()


def run_process(
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> None:
    ProcessApp(input_path=input_path, output_dir=output_dir).run()


__all__ = ["ProcessApp", "SetupApp", "run_process", "run_setup"]
