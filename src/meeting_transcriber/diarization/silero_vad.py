"""Silero VAD wrapper used to build NeMo's external VAD manifest.

Silero VAD is language-agnostic and its model ships inside the ``silero-vad``
wheel, so no checkpoint has to be downloaded at runtime.
"""

from __future__ import annotations

from pathlib import Path

from ..config import DiarizationConfig
from ..errors import BackendUnavailableError


def run_silero_vad(
    wav_path: str | Path,
    cfg: DiarizationConfig,
) -> list[tuple[float, float]]:
    """Return speech intervals in seconds."""

    try:
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError as exc:
        raise BackendUnavailableError(
            "silero-vad is not installed. Install the diarization extras: "
            "pip install -r requirements/diarization.txt"
        ) from exc

    audio = _read_16k_mono_wav(wav_path)
    model = load_silero_vad()
    timestamps = get_speech_timestamps(
        audio,
        model,
        sampling_rate=16000,
        threshold=cfg.vad_threshold,
        min_speech_duration_ms=cfg.vad_min_speech_ms,
        min_silence_duration_ms=cfg.vad_min_silence_ms,
        speech_pad_ms=cfg.vad_speech_pad_ms,
        return_seconds=True,
    )
    intervals: list[tuple[float, float]] = []
    for item in timestamps:
        start = float(item["start"])
        end = float(item["end"])
        if end > start:
            intervals.append((round(start, 3), round(end, 3)))
    return intervals


def _read_16k_mono_wav(wav_path: str | Path):
    """Read our own 16 kHz mono PCM WAV without torchaudio/torchcodec.

    Silero's ``read_audio`` helper requires an extra audio backend, but the VAD
    model itself only needs a float32 tensor, so a small ``wave`` reader keeps
    the dependency set minimal and fully offline.
    """

    import wave

    import numpy as np
    import torch

    with wave.open(str(wav_path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getframerate() != 16000:
            raise BackendUnavailableError(
                f"{wav_path}: expected 16 kHz mono WAV for VAD"
            )
        if handle.getsampwidth() != 2:
            raise BackendUnavailableError(f"{wav_path}: expected 16-bit PCM WAV for VAD")
        frames = handle.readframes(handle.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(samples)
