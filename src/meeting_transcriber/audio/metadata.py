"""Audio probing, hashing and validation via FFprobe."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from ..errors import AudioError
from ..types import AudioMetadata


def require_executable(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise AudioError(
            f"required executable '{name}' was not found in PATH. "
            "Install FFmpeg (see scripts/install_host_dependencies.sh)."
        )
    return path


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_audio(path: Path, ffprobe: str = "ffprobe", audio_stream: int = 0) -> AudioMetadata:
    """Read stream metadata with ffprobe; raises AudioError on unreadable input.

    ``audio_stream`` selects an audio stream by its 0-based position among all
    audio streams (FFmpeg's ``0:a:N``). Video files are fully supported: only
    the selected audio track is used.
    """

    require_executable(ffprobe)
    if not path.is_file():
        raise AudioError(f"audio file not found: {path}")
    if audio_stream < 0:
        raise AudioError(f"audio stream index must be >= 0, got {audio_stream}")

    command = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise AudioError(f"ffprobe failed for {path}: {completed.stderr.strip()}")

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise AudioError(f"ffprobe returned invalid JSON for {path}: {exc}") from exc

    audio_streams = [s for s in payload.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio_streams:
        raise AudioError(f"{path} contains no audio stream (video without an audio track?)")
    if audio_stream >= len(audio_streams):
        raise AudioError(
            f"{path} has {len(audio_streams)} audio stream(s); "
            f"requested stream {audio_stream}. Use --audio-stream with a value "
            f"between 0 and {len(audio_streams) - 1}."
        )
    stream = audio_streams[audio_stream]
    video_streams = [s for s in payload.get("streams", []) if s.get("codec_type") == "video"]
    fmt = payload.get("format", {})

    duration = _as_float(stream.get("duration")) or _as_float(fmt.get("duration"))
    if duration is None or duration <= 0:
        raise AudioError(f"could not determine a positive duration for {path}")

    metadata = AudioMetadata(
        filename=path.name,
        path=str(path.resolve()),
        duration=duration,
        sample_rate=_as_int(stream.get("sample_rate")),
        channels=_as_int(stream.get("channels")),
        codec=stream.get("codec_name"),
        format=fmt.get("format_name"),
        bit_rate=_as_int(fmt.get("bit_rate")),
        has_video=bool(video_streams),
        video_codec=video_streams[0].get("codec_name") if video_streams else None,
        audio_streams=len(audio_streams),
        audio_stream_index=audio_stream,
    )
    if metadata.channels and metadata.channels > 2:
        metadata.warnings.append(
            f"input has {metadata.channels} channels; a mono mixdown is used for models"
        )
    if len(audio_streams) > 1:
        metadata.warnings.append(
            f"input has {len(audio_streams)} audio streams; using stream {audio_stream}"
        )
    return metadata


def _as_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
