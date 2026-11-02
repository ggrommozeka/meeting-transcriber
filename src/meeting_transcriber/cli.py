"""Command line interface."""

from __future__ import annotations

import argparse
import json
import os
import site
import sys
from pathlib import Path

from . import __version__
from .audio.metadata import probe_audio
from .audio.preprocess import extract_audio_track
from .config import AppConfig, load_config
from .errors import ConfigError, TranscriberError
from .models import download_models, list_models, verify_models
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meeting-transcriber",
        description="Fully local meeting transcription with speaker diarization.",
    )
    parser.add_argument("--version", action="version", version=f"meeting-transcriber {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    transcribe = subparsers.add_parser("transcribe", help="Transcribe an audio file")
    transcribe.add_argument("audio", type=Path)
    _add_config_options(transcribe)
    transcribe.add_argument("--language", default=None, help="ASR language (default: ru)")
    transcribe.add_argument("--output-dir", default=None)
    transcribe.add_argument("--profile", default=None, help="Terminology profile (single)")
    transcribe.add_argument("--num-speakers", type=int, default=None)
    transcribe.add_argument("--max-speakers", type=int, default=None)
    transcribe.add_argument("--skip-diarization", action="store_true")
    transcribe.add_argument("--skip-llm", action="store_true")
    transcribe.add_argument("--speaker-map", type=Path, default=None)
    transcribe.add_argument("--device", choices=["cuda", "cpu", "auto"], default=None)
    transcribe.add_argument(
        "--audio-stream",
        type=int,
        default=None,
        help="0-based audio stream index for multi-track video (FFmpeg 0:a:N)",
    )
    transcribe.add_argument("--force", action="store_true", help="Ignore cached stages and overwrite outputs")

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
        command.add_argument("--only", choices=["all", "asr", "diarization", "llm"], default="all")
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
        return download_models(
            _manifest_path(cfg), cfg.models_path(), only=args.only, force=args.force
        )
    if args.command == "verify-models":
        return verify_models(_manifest_path(cfg), cfg.models_path(), only=args.only)
    if args.command == "list-models":
        list_models(_manifest_path(cfg), cfg.models_path())
        return 0

    if args.command == "transcribe":
        return _transcribe(args, cfg)
    raise ConfigError(f"unknown command: {args.command}")


def _transcribe(args: argparse.Namespace, cfg: AppConfig) -> int:
    if args.language:
        cfg.language = args.language
    if args.audio_stream is not None:
        cfg.audio.audio_stream = args.audio_stream
    pipeline = Pipeline(
        cfg,
        force=args.force,
        skip_diarization=args.skip_diarization,
        skip_llm=args.skip_llm,
        num_speakers=args.num_speakers,
        max_speakers=args.max_speakers,
        device=args.device,
        profile=args.profile,
        output_dir=args.output_dir,
        speaker_map=str(args.speaker_map) if args.speaker_map else None,
    )
    result = pipeline.run(args.audio)
    transcript = result.transcript
    print(f"meeting_id: {transcript.meeting_id}")
    print(f"language:   {transcript.language}")
    print(f"duration:   {transcript.source_audio.duration:.1f}s")
    print(f"segments:   {len(transcript.segments)}")
    print(f"speakers:   {len(transcript.speaker_ids)}")
    if transcript.run.vram_peak_mb is not None:
        print(f"peak VRAM:  {transcript.run.vram_peak_mb:.0f} MiB")
    for fmt, path in result.outputs.items():
        print(f"{fmt:9}  {path}")
    for warning in transcript.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


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
    if args.map:
        applied = apply_speaker_names(transcript, load_speaker_map(args.map))
        print(f"applied {applied} speaker name(s)")

    work_dir = cfg.resolve(cfg.paths.work_dir) / transcript.meeting_id
    report, warnings = generate_meeting_report(
        transcript,
        cfg.llm,
        work_dir=work_dir,
        progress=lambda message: print(message, file=sys.stderr),
    )
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if report is None:
        return 1

    stem = transcript_path.with_suffix("")
    report_json = stem.with_suffix(".report.json")
    report_md = stem.with_suffix(".report.md")
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_md.write_text(render_report_markdown(report), encoding="utf-8")
    write_markdown(transcript, stem.with_suffix(".markdown"), report)
    print(f"report json: {report_json}")
    print(f"report md:   {report_md}")
    return 0


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
