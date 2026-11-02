"""Audio preprocessing: video/audio track extraction, mono 16 kHz conversion,
and chunk planning.

FFmpeg is used in two clearly separated steps:

1. **Track extraction** (video input, ``audio.keep_original: true``): the
   selected audio stream is copied losslessly into ``audio_original.mka``
   (stream copy, FLAC fallback). This is the "cut the audio track out of the
   meeting video" step.
2. **Model conversion**: the extracted track (or the original audio file) is
   resampled to 16 kHz mono PCM WAV and split into overlapping chunks.

The source file is never modified.
"""

from __future__ import annotations

import json
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from ..config import AudioConfig
from ..errors import AudioError
from ..types import AudioMetadata
from .metadata import file_sha256, probe_audio, require_executable

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
EXTRACTED_AUDIO_NAME = "audio_original.mka"


@dataclass
class AudioChunk:
    index: int
    start: float
    end: float
    path: Path

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class PreparedAudio:
    source: Path
    wav_path: Path
    metadata: AudioMetadata
    chunks: list[AudioChunk]
    original_audio: Path | None = None


def plan_chunks(duration: float, chunk_seconds: float, overlap_seconds: float) -> list[tuple[float, float]]:
    """Return ``(start, end)`` windows covering ``duration`` with overlap.

    Consecutive windows overlap by ``overlap_seconds``. The final window is
    clamped to the end of the recording. When the recording fits in a single
    chunk the result is ``[(0.0, duration)]``.
    """

    if duration <= 0:
        raise ValueError("duration must be positive")
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    if overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be in [0, chunk_seconds)")

    windows: list[tuple[float, float]] = []
    start = 0.0
    step = chunk_seconds - overlap_seconds
    while True:
        end = min(start + chunk_seconds, duration)
        windows.append((round(start, 3), round(end, 3)))
        if end >= duration:
            break
        start = round(start + step, 3)
    return windows


def build_extract_audio_command(
    ffmpeg: str,
    source: str | Path,
    destination: str | Path,
    audio_stream: int = 0,
    codec: str = "copy",
    container_format: str | None = None,
) -> list[str]:
    """FFmpeg command that extracts one audio stream from a video/container.

    ``codec="copy"`` avoids re-encoding. ``codec="flac"`` is used as a fallback
    when the source codec cannot be stored in the Matroska audio container.
    ``container_format`` is passed as an explicit ``-f`` so temporary file names
    (for atomic writes) never confuse FFmpeg's muxer selection.
    """

    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-map",
        f"0:a:{audio_stream}",
        "-c:a",
        codec,
    ]
    if container_format:
        command += ["-f", container_format]
    command.append(str(destination))
    return command


def build_convert_to_wav_command(
    ffmpeg: str,
    source: str | Path,
    destination: str | Path,
    audio_stream: int = 0,
) -> list[str]:
    """FFmpeg command that converts a track to 16 kHz mono PCM WAV."""

    return [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-map",
        f"0:a:{audio_stream}",
        "-ac",
        str(TARGET_CHANNELS),
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]


def extract_audio_track(
    source: str | Path,
    destination: str | Path,
    audio_stream: int = 0,
    ffmpeg: str = "ffmpeg",
) -> Path:
    """Copy one audio stream into ``destination`` without re-encoding.

    Falls back to FLAC if the stream copy cannot be muxed. The destination is
    written atomically; a failed first attempt never leaves a partial file.
    """

    require_executable(ffmpeg)
    src = Path(source).expanduser().resolve()
    dst = Path(destination).expanduser().resolve()
    if not src.is_file():
        raise AudioError(f"audio/video file not found: {src}")
    if src == dst:
        raise AudioError("source and destination for audio extraction must differ")
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Keep a real container extension on the temporary file: FFmpeg refuses to
    # infer a muxer from names like "audio_original.mka.part".
    tmp = dst.with_name(f"{dst.stem}.part{dst.suffix}")

    errors: list[str] = []
    suffix = dst.suffix.lower()
    if suffix == ".flac":
        codecs = ["flac"]
        container_format = "flac"
    elif suffix == ".wav":
        codecs = ["pcm_s16le"]
        container_format = "wav"
    else:
        codecs = ["copy", "flac"]
        container_format = "matroska"
    for codec in codecs:
        tmp.unlink(missing_ok=True)
        command = build_extract_audio_command(
            ffmpeg, src, tmp, audio_stream, codec, container_format=container_format
        )
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0:
            tmp.replace(dst)
            return dst
        errors.append(f"{codec}: {completed.stderr.strip() or 'empty output'}")
        tmp.unlink(missing_ok=True)

    raise AudioError(
        f"FFmpeg could not extract audio stream {audio_stream} from {src}: "
        + " | ".join(errors)
    )


def preprocess_audio(
    input_path: str | Path,
    work_dir: str | Path,
    audio_cfg: AudioConfig,
    chunk_seconds: float = 900.0,
    overlap_seconds: float = 15.0,
) -> PreparedAudio:
    """Probe the source, extract its audio track and create a 16 kHz mono WAV.

    Accepts any container FFmpeg understands, including MP4/MKV/MOV/WebM/AVI
    meeting recordings. When ``audio.keep_original`` is true and the input has
    a video stream, the selected audio track is copied losslessly to
    ``work/<meeting>/audio_original.mka`` before conversion.
    """

    source = Path(input_path).expanduser().resolve()
    metadata = probe_audio(
        source, ffprobe=audio_cfg.ffprobe, audio_stream=audio_cfg.audio_stream
    )
    metadata.sha256 = file_sha256(source)

    work = Path(work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    wav_path = work / "audio_16k_mono.wav"
    chunks_dir = work / "chunks"
    provenance_path = work / "audio_source.json"

    # Reuse derivatives only if they were produced from this exact source and
    # audio stream. This protects against a new file reusing the same meeting id.
    stale = _is_stale_provenance(provenance_path, wav_path, metadata.sha256, metadata.audio_stream_index)
    if stale:
        wav_path.unlink(missing_ok=True)
        (work / EXTRACTED_AUDIO_NAME).unlink(missing_ok=True)
        if chunks_dir.is_dir():
            for chunk in chunks_dir.glob("chunk_*.wav"):
                chunk.unlink(missing_ok=True)

    original_audio: Path | None = None
    wav_source = source
    wav_stream = metadata.audio_stream_index

    if audio_cfg.keep_original and metadata.has_video:
        original_audio = work / EXTRACTED_AUDIO_NAME
        if stale or not original_audio.exists():
            extract_audio_track(
                source,
                original_audio,
                audio_stream=metadata.audio_stream_index,
                ffmpeg=audio_cfg.ffmpeg,
            )
        metadata.extracted_audio = str(original_audio)
        wav_source = original_audio
        # The extracted container has exactly one audio track.
        wav_stream = 0

    if not wav_path.exists():
        ffmpeg = require_executable(audio_cfg.ffmpeg)
        command = build_convert_to_wav_command(ffmpeg, wav_source, wav_path, wav_stream)
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0 or not wav_path.exists():
            raise AudioError(
                f"FFmpeg failed to convert {wav_source}: "
                f"{completed.stderr.strip() or 'unknown error'}"
            )

    _validate_wav(wav_path)
    chunks = _write_chunks(
        wav_path,
        chunks_dir,
        plan_chunks(metadata.duration, chunk_seconds, overlap_seconds),
    )
    provenance_path.write_text(
        json.dumps(
            {
                "source": str(source),
                "sha256": metadata.sha256,
                "audio_stream": metadata.audio_stream_index,
                "wav_path": str(wav_path),
                "original_audio": str(original_audio) if original_audio else None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return PreparedAudio(
        source=source,
        wav_path=wav_path,
        metadata=metadata,
        chunks=chunks,
        original_audio=original_audio,
    )


def _is_stale_provenance(
    provenance_path: Path, wav_path: Path, source_sha256: str, audio_stream: int
) -> bool:
    if not wav_path.exists():
        return False
    if not provenance_path.is_file():
        return True
    try:
        data = json.loads(provenance_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return True
    return data.get("sha256") != source_sha256 or data.get("audio_stream") != audio_stream


def _validate_wav(path: Path) -> None:
    try:
        with wave.open(str(path), "rb") as handle:
            if handle.getnchannels() != TARGET_CHANNELS:
                raise AudioError(f"{path}: expected mono WAV")
            if handle.getframerate() != TARGET_SAMPLE_RATE:
                raise AudioError(f"{path}: expected {TARGET_SAMPLE_RATE} Hz WAV")
            if handle.getnframes() <= 0:
                raise AudioError(f"{path}: WAV contains no frames")
    except wave.Error as exc:
        raise AudioError(f"{path}: invalid WAV file: {exc}") from exc


def _write_chunks(wav_path: Path, chunk_dir: Path, windows: list[tuple[float, float]]) -> list[AudioChunk]:
    with wave.open(str(wav_path), "rb") as handle:
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
        total_duration = frame_count / sample_rate

        if len(windows) == 1 and windows[0][0] == 0.0:
            return [AudioChunk(index=0, start=0.0, end=total_duration, path=wav_path)]

        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunks: list[AudioChunk] = []
        for index, (start, end) in enumerate(windows):
            chunk_path = chunk_dir / f"chunk_{index:04d}.wav"
            start_frame = int(round(start * sample_rate))
            end_frame = min(int(round(end * sample_rate)), frame_count)
            expected_frames = max(0, end_frame - start_frame)
            if not chunk_path.exists() or _frame_count(chunk_path) != expected_frames:
                handle.setpos(start_frame)
                frames = handle.readframes(expected_frames)
                with wave.open(str(chunk_path), "wb") as out:
                    out.setnchannels(TARGET_CHANNELS)
                    out.setsampwidth(2)
                    out.setframerate(sample_rate)
                    out.writeframes(frames)
            chunks.append(AudioChunk(index=index, start=start, end=end, path=chunk_path))
    return chunks


def _frame_count(path: Path) -> int:
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes()
    except (wave.Error, OSError):
        return -1
