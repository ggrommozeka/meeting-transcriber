"""Command line interface."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import site
import sys
from pathlib import Path

from . import __version__
from .audio.metadata import probe_audio
from .audio.preprocess import extract_audio_track
from .config import AppConfig, load_config
from .errors import ConfigError, TranscriberError
from .models import download_models, iter_groups, list_models, load_manifest, verify_models
from .output.json_writer import write_json
from .output.markdown_writer import write_markdown
from .output.srt_writer import write_srt
from .pipeline import persistence
from .pipeline.runner import Pipeline
from .postprocessing.meeting_report import (
    generate_meeting_report,
    render_report_markdown,
)
from .speakers import (
    apply_speaker_names,
    collect_speaker_stats,
    load_speaker_map,
    write_speaker_template,
)
from .terminology.codebase_import import scan_cpp_repository, write_glossary
from .ui import (
    DownloadProgress,
    StageProgress,
    banner,
    choose_model_groups,
    console,
    err_console,
    result_panel,
)
from .visual import load_visual_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meeting-transcriber",
        description="Fully local meeting transcription with speaker diarization.",
    )
    parser.add_argument("--version", action="version", version=f"meeting-transcriber {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    transcribe = subparsers.add_parser("transcribe", help="Transcribe an audio file")
    transcribe.add_argument("audio", nargs="?", type=Path, help="Input audio or video file")
    transcribe.add_argument(
        "-i", "--input", dest="input", type=Path, default=None, help="Input audio or video file"
    )
    _add_config_options(transcribe)
    transcribe.add_argument("--language", default=None, help="ASR language (default: ru)")
    transcribe.add_argument("--output-dir", default=None)
    transcribe.add_argument("--profile", default=None, help="Terminology profile (single)")
    transcribe.add_argument("--num-speakers", type=int, default=None)
    transcribe.add_argument("--max-speakers", type=int, default=None)
    transcribe.add_argument("--skip-diarization", action="store_true")
    transcribe.add_argument("--skip-llm", action="store_true")
    transcribe.add_argument(
        "--llm",
        action="store_true",
        help="Generate the LLM meeting report (requires the local llama-server)",
    )
    transcribe.add_argument(
        "--visual",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable optional OCR of slides/code for report context (off by default)",
    )
    transcribe.add_argument(
        "--keep-frames",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep frames of code cards next to the report",
    )
    transcribe.add_argument(
        "--ocr-backend",
        choices=["tesseract"],
        default=None,
        help="Local OCR backend for visual analysis",
    )
    transcribe.add_argument("--speaker-map", type=Path, default=None)
    transcribe.add_argument(
        "--report-style",
        choices=["asd-ste100", "plain"],
        default=None,
        help="Report profile: short ASD-STE100 or full plain",
    )
    transcribe.add_argument("--device", choices=["cuda", "cpu", "auto"], default=None)
    transcribe.add_argument(
        "--audio-stream",
        type=int,
        default=None,
        help="0-based audio stream index for multi-track video (FFmpeg 0:a:N)",
    )
    transcribe.add_argument("--force", action="store_true", help="Ignore cached stages and overwrite outputs")

    video = subparsers.add_parser(
        "transcribe-video",
        help="Transcribe a video and OCR slides/code shown on screen",
    )
    video.add_argument("audio", nargs="?", type=Path, help="Input video file")
    video.add_argument(
        "-i", "--input", dest="input", type=Path, default=None, help="Input video file"
    )
    _add_config_options(video)
    video.add_argument("--output-dir", default=None)
    video.add_argument("--language", default=None, help="ASR language (default: ru)")
    video.add_argument("--num-speakers", type=int, default=None)
    video.add_argument("--max-speakers", type=int, default=None)
    video.add_argument("--device", choices=["cuda", "cpu", "auto"], default=None)
    video.add_argument(
        "--keep-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep frames of code cards next to the transcript",
    )
    video.add_argument("--force", action="store_true", help="Ignore cached stages")

    extract = subparsers.add_parser(
        "extract-audio", help="Extract the audio track from a video/container with FFmpeg"
    )
    extract.add_argument("media", type=Path)
    extract.add_argument("--output", type=Path, default=None)
    extract.add_argument("--audio-stream", type=int, default=None)
    extract.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    _add_config_options(extract)

    for name, help_text in (
        ("verify-models", "Verify local model files and SHA-256 checksums"),
        ("list-models", "List configured models and whether they are present"),
        ("download-models", "Download model weights (explicit host-side action)"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        _add_config_options(command)
        command.add_argument("--only", choices=["all", "asr", "diarization", "llm"], default=None)
        if name == "download-models":
            command.add_argument("--force", action="store_true", help="Re-download existing files")

    validate = subparsers.add_parser("validate-config", help="Validate configuration and print it")
    _add_config_options(validate)

    speakers = subparsers.add_parser(
        "speakers", help="Show speaker statistics and write a naming template"
    )
    speakers.add_argument("transcript", type=Path, help="Path to the transcript JSON")
    speakers.add_argument("--template", type=Path, default=None)
    speakers.add_argument("--force", action="store_true", help="Overwrite the template")

    label = subparsers.add_parser(
        "label-speakers", help="Apply display names to an existing transcript"
    )
    label.add_argument("transcript", type=Path, help="Path to the transcript JSON")
    label.add_argument("--map", type=Path, required=True, help="Speaker names YAML")

    report = subparsers.add_parser(
        "report", help="Generate meeting minutes from an existing transcript"
    )
    report.add_argument("transcript", type=Path, help="Path to the transcript JSON")
    report.add_argument("--map", type=Path, default=None, help="Speaker names YAML")
    report.add_argument(
        "--visual",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use cached visual OCR cards from the work directory",
    )
    report.add_argument(
        "--report-style",
        choices=["asd-ste100", "plain"],
        default=None,
        help="Report profile: short ASD-STE100 or full plain",
    )
    _add_config_options(report)

    glossary = subparsers.add_parser("build-glossary", help="Extract terminology from a code base")
    glossary.add_argument("project_path", type=Path)
    glossary.add_argument("--output", type=Path, default=None)
    glossary.add_argument("--profile", default=None)
    glossary.add_argument("--max-files", type=int, default=5000)

    return parser


def _add_config_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="YAML configuration file")
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Override paths.models_dir (weights directory)",
    )


def _load(args: argparse.Namespace) -> AppConfig:
    overrides = None
    if getattr(args, "models_dir", None):
        overrides = {"paths": {"models_dir": args.models_dir}}
    return load_config(getattr(args, "config", None), overrides=overrides)


def _manifest_path(cfg: AppConfig) -> Path:
    candidates = [cfg.resolve("models/manifest.yaml"), cfg.models_path("manifest.yaml")]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ConfigError(
        "models/manifest.yaml not found. Set paths.models_dir or keep the manifest in models/."
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "transcribe":
        _ensure_nvidia_library_path()
    try:
        return _dispatch(args)
    except TranscriberError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


def _ensure_nvidia_library_path() -> None:
    """Re-exec with pip-installed CUDA libraries on ``LD_LIBRARY_PATH``.

    ``ctranslate2`` dlopen()s cuBLAS/cuDNN by SONAME. Wheels from
    ``nvidia-cublas-cu12``/``nvidia-cudnn-cu12`` live in site-packages and are
    not on the loader path by default. Re-exec is the only reliable way to make
    the dynamic loader see them, so users do not need to export anything.
    """

    if os.environ.get("_MEETING_TRANSCRIBER_CUDA_ENV") == "1":
        return
    try:
        roots = {Path(path) / "nvidia" for path in site.getsitepackages()}
        lib_dirs = sorted(
            str(lib_dir)
            for root in roots
            if root.is_dir()
            for lib_dir in root.glob("*/lib")
            if lib_dir.is_dir()
        )
    except Exception:
        return
    if not lib_dirs:
        return

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing] if existing else []))
    os.environ["_MEETING_TRANSCRIBER_CUDA_ENV"] = "1"
    os.execv(sys.executable, [sys.executable, "-m", "meeting_transcriber", *sys.argv[1:]])


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "build-glossary":
        return _build_glossary(args)
    if args.command == "speakers":
        return _speakers(args)
    if args.command == "label-speakers":
        return _label_speakers(args)

    cfg = _load(args)

    if args.command == "validate-config":
        print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))
        print("configuration is valid", file=sys.stderr)
        return 0

    if args.command == "extract-audio":
        return _extract_audio(args, cfg)

    if args.command == "report":
        return _report(args, cfg)

    if args.command == "download-models":
        return _download_models(args, cfg)
    if args.command == "verify-models":
        return verify_models(_manifest_path(cfg), cfg.models_path(), only=args.only or "all")
    if args.command == "list-models":
        list_models(_manifest_path(cfg), cfg.models_path())
        return 0

    if args.command == "transcribe":
        return _transcribe(args, cfg)
    if args.command == "transcribe-video":
        return _transcribe_video(args, cfg)
    raise ConfigError(f"unknown command: {args.command}")


def _download_models(args: argparse.Namespace, cfg: AppConfig) -> int:
    manifest_path = _manifest_path(cfg)
    models_dir = cfg.models_path()
    if args.only:
        selected = [args.only]
    elif sys.stdin.isatty():
        manifest = load_manifest(manifest_path)
        groups = []
        for group in iter_groups(manifest):
            total = sum(item.size or 0 for item in group.files)
            groups.append(
                {
                    "key": group.key,
                    "name": group.name,
                    "size": f"{total / 1024**3:.1f} GiB" if total else "—",
                    "present": all(
                        (models_dir / group.target_dir / item.name).is_file()
                        for item in group.files
                    ),
                }
            )
        selected = choose_model_groups(groups)
        if not selected:
            console.print("[muted]Загрузка отменена.[/muted]")
            return 0
    else:
        selected = ["all"]

    banner("meeting-transcriber · модели", f"[muted]{models_dir}[/muted]")
    progress = DownloadProgress() if err_console.is_terminal else None
    code = 0
    try:
        for only in selected:
            code |= download_models(
                manifest_path,
                models_dir,
                only=only,
                force=args.force,
                progress=progress,
            )
    finally:
        if progress is not None:
            progress.close()
    if code == 0:
        result_panel("Модели готовы", [("Каталог", str(models_dir))])
    return code


def _transcribe(args: argparse.Namespace, cfg: AppConfig) -> int:
    input_path = getattr(args, "input", None) or getattr(args, "audio", None)
    if input_path is None:
        raise ConfigError("input file is required: pass -i FILE or a positional path")
    if getattr(args, "language", None):
        cfg.language = args.language
    if getattr(args, "audio_stream", None) is not None:
        cfg.audio.audio_stream = args.audio_stream
    if getattr(args, "report_style", None) is not None:
        cfg.llm.report_style = args.report_style
    if getattr(args, "visual", None) is not None:
        cfg.visual.enabled = args.visual
    if getattr(args, "keep_frames", None) is not None:
        cfg.visual.keep_code_frames = args.keep_frames
    if getattr(args, "ocr_backend", None) is not None:
        cfg.visual.backend = args.ocr_backend

    output_dir = getattr(args, "output_dir", None)
    if output_dir is None:
        output_dir = str(Path(input_path).expanduser().resolve().parent)
    banner(
        "meeting-transcriber",
        f"[bold]{Path(input_path).name}[/bold] → [muted]{output_dir}[/muted]",
    )
    with StageProgress(show=err_console.is_terminal) as progress:
        pipeline = Pipeline(
            cfg,
            force=getattr(args, "force", False),
            skip_diarization=getattr(args, "skip_diarization", False),
            skip_llm=(
                not getattr(args, "llm", False)
                or getattr(args, "skip_llm", False)
            ),
            num_speakers=getattr(args, "num_speakers", None),
            max_speakers=getattr(args, "max_speakers", None),
            device=getattr(args, "device", None),
            profile=getattr(args, "profile", None),
            output_dir=output_dir,
            speaker_map=(
                str(args.speaker_map) if getattr(args, "speaker_map", None) else None
            ),
            progress=progress,
        )
        result = pipeline.run(input_path)
    transcript = result.transcript
    rows = [
        ("Файл", Path(input_path).name),
        ("Язык", transcript.language),
        ("Длительность", f"{(transcript.source_audio.duration or 0) / 60:.1f} мин"),
        ("Сегменты", str(len(transcript.segments))),
        ("Спикеры", str(len(transcript.speaker_ids))),
    ]
    if transcript.run.vram_peak_mb is not None:
        rows.append(("Пик VRAM", f"{transcript.run.vram_peak_mb:.0f} MiB"))
    if result.visual and result.visual.cards:
        code_count = sum(1 for card in result.visual.cards if card.kind == "code")
        rows.append(("Карточки экрана", f"{len(result.visual.cards)} ({code_count} с кодом)"))
    result_panel("Транскрипция готова", rows)
    for fmt, path in result.outputs.items():
        console.print(f"  [brand]{fmt:<14}[/brand] [muted]{path}[/muted]")
    for warning in transcript.warnings:
        err_console.print(f"[warn]warning:[/warn] {warning}")
    return 0


def _transcribe_video(args: argparse.Namespace, cfg: AppConfig) -> int:
    args.visual = True
    args.skip_llm = True
    args.skip_diarization = False
    args.speaker_map = None
    args.profile = None
    args.report_style = None
    args.audio_stream = None
    args.ocr_backend = None
    return _transcribe(args, cfg)


def _speakers(args: argparse.Namespace) -> int:
    transcript_path = args.transcript.expanduser().resolve()
    transcript = persistence.load_transcript(transcript_path)
    stats = collect_speaker_stats(transcript)
    if not stats:
        print("no speaker-labelled segments in the transcript")
        return 0

    print(f"{'speaker':12} {'talk':>7} {'segments':>8}  name")
    for entry in stats:
        name = next(
            (
                speaker.display_name
                for speaker in transcript.speakers
                if speaker.speaker_id == entry.speaker_id
            ),
            None,
        )
        print(
            f"{entry.speaker_id:12} {entry.talk_time_minutes:6.1f}m "
            f"{entry.segments:8d}  {name or '-'}"
        )
        for sample in entry.samples:
            print(f"    · {sample}")

    template = args.template or transcript_path.with_name(
        f"{transcript_path.stem}.speakers.yaml"
    )
    created = write_speaker_template(transcript, template, overwrite=args.force)
    print(f"\ntemplate: {created}")
    if created.exists():
        print("fill in the 'name' fields, then run:")
        print(
            f"  meeting-transcriber label-speakers {transcript_path.name} "
            f"--map {created.name}"
        )
    return 0


def _label_speakers(args: argparse.Namespace) -> int:
    transcript_path = args.transcript.expanduser().resolve()
    transcript = persistence.load_transcript(transcript_path)
    mapping = load_speaker_map(args.map)
    applied = apply_speaker_names(transcript, mapping)
    if not applied:
        print("warning: no matching speakers were found in the map", file=sys.stderr)

    stem = transcript_path.with_suffix("")
    report_path = stem.with_suffix(".report.json")
    report = None
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))

    write_json(transcript, stem.with_suffix(".json"))
    write_markdown(transcript, stem.with_suffix(".markdown"), report)
    write_srt(transcript, stem.with_suffix(".srt"))
    print(f"applied {applied} name(s)")
    print(f"json:     {stem.with_suffix('.json')}")
    print(f"markdown: {stem.with_suffix('.markdown')}")
    print(f"srt:      {stem.with_suffix('.srt')}")
    if report_path.is_file():
        print("note: existing report.md keeps the names used when it was generated")
    return 0


def _report(args: argparse.Namespace, cfg: AppConfig) -> int:
    transcript_path = args.transcript.expanduser().resolve()
    transcript = persistence.load_transcript(transcript_path)
    if args.report_style is not None:
        cfg.llm.report_style = args.report_style
    if args.map:
        applied = apply_speaker_names(transcript, load_speaker_map(args.map))
        print(f"applied {applied} speaker name(s)")

    work_dir = cfg.resolve(cfg.paths.work_dir) / transcript.meeting_id
    banner("meeting-transcriber · отчёт", f"[bold]{transcript.meeting_id}[/bold]")
    cards = []
    if args.visual is not False:
        visual = load_visual_result(work_dir)
        if visual is not None:
            cards = visual.cards
            if cards:
                console.print(f"[muted]visual: {len(cards)} cached screen card(s)[/muted]")
    report, warnings = generate_meeting_report(
        transcript,
        cfg.llm,
        work_dir=work_dir,
        progress=lambda message: err_console.print(f"[muted]{message}[/muted]"),
        code_cards=cards,
        visual_context_chars=cfg.visual.report_context_chars,
        visual_align_window=cfg.visual.align_window_seconds,
    )
    for warning in warnings:
        err_console.print(f"[warn]warning:[/warn] {warning}")
    if report is None:
        return 1

    stem = transcript_path.with_suffix("")
    report_json = stem.with_suffix(".report.json")
    report_md = stem.with_suffix(".report.md")
    frames_rel = f"{stem.name}.frames"
    frames_dir = stem.parent / frames_rel
    copied_frames = _copy_frames(report, work_dir / "frames", frames_dir)
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_md.write_text(
        render_report_markdown(report, frames_dir=frames_rel), encoding="utf-8"
    )
    write_markdown(transcript, stem.with_suffix(".markdown"), report)
    rows = [("JSON", str(report_json)), ("Markdown", str(report_md))]
    if copied_frames:
        rows.append(("Кадры", str(frames_dir)))
    result_panel("Отчёт готов", rows)
    return 0


def _copy_frames(report: dict, source_dir: Path, target_dir: Path) -> bool:
    frames = {
        str(card.get("frame"))
        for card in report.get("code_cards") or []
        if isinstance(card, dict) and card.get("frame")
    }
    copied = False
    for name in frames:
        source = source_dir / name
        if source.is_file():
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target_dir / name)
            copied = True
    return copied


def _extract_audio(args: argparse.Namespace, cfg: AppConfig) -> int:
    source = args.media.expanduser().resolve()
    stream = args.audio_stream if args.audio_stream is not None else cfg.audio.audio_stream
    metadata = probe_audio(source, ffprobe=cfg.audio.ffprobe, audio_stream=stream)

    output = args.output
    if output is None:
        output = cfg.resolve(cfg.paths.output_dir) / f"{source.stem}.mka"
    output = output.expanduser().resolve()
    if output.exists() and not args.force:
        raise ConfigError(f"output already exists: {output}. Pass --force to overwrite.")
    if output.suffix.lower() not in {".mka", ".flac", ".wav"}:
        raise ConfigError(
            f"unsupported extraction container '{output.suffix}'. Use .mka (recommended, "
            "lossless stream copy), .flac or .wav."
        )

    path = extract_audio_track(
        source,
        output,
        audio_stream=metadata.audio_stream_index,
        ffmpeg=cfg.audio.ffmpeg,
    )
    kind = "video" if metadata.has_video else "audio"
    print(f"source:        {source} ({kind}, {metadata.audio_streams} audio stream(s))")
    print(f"video codec:   {metadata.video_codec or '-'}")
    print(f"audio stream:  {metadata.audio_stream_index} ({metadata.codec})")
    print(f"duration:      {metadata.duration:.1f}s")
    print(f"extracted to:  {path}")
    return 0


def _build_glossary(args: argparse.Namespace) -> int:
    result = scan_cpp_repository(args.project_path, max_files=args.max_files)
    output = args.output or Path("config/terminology/generated.yaml")
    write_glossary(result.terms, output, profile=args.profile)
    print(f"scanned {result.files_scanned} file(s)")
    print(f"extracted {len(result.terms)} term(s) -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
