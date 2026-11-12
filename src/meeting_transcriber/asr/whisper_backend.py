"""faster-whisper (CTranslate2) backend.

The model is always loaded from a local directory. Passing a Hugging Face model
id is rejected before CTranslate2 is imported, so runtime cannot silently reach
the network.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

from ..audio.preprocess import AudioChunk
from ..errors import BackendUnavailableError, ModelNotFoundError
from ..types import Segment, Word
from .base import ASRResult, Transcriber

REQUIRED_FILES = ("model.bin", "config.json", "tokenizer.json", "vocabulary.json")


class WhisperBackend(Transcriber):
    name = "faster-whisper"

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cuda",
        compute_type: str = "float16",
        cpu_threads: int = 4,
        beam_size: int = 5,
        word_timestamps: bool = True,
        vad_filter: bool = True,
        vad_min_silence_ms: int = 500,
        condition_on_previous_text: bool = False,
        local_files_only: bool = True,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser()
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.beam_size = beam_size
        self.word_timestamps = word_timestamps
        self.vad_filter = vad_filter
        self.vad_min_silence_ms = vad_min_silence_ms
        self.condition_on_previous_text = condition_on_previous_text
        self.local_files_only = local_files_only
        self._model: Any = None
        self._detected_language: str | None = None

    # -- lifecycle -----------------------------------------------------------
    def load(self) -> None:
        self._validate_model_directory()
        if self._model is not None:
            return

        # Defensive belt-and-braces: a local path is used below, but if some
        # transitive dependency consults the Hub it must not succeed.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise BackendUnavailableError(
                "faster-whisper is not installed. Install the ASR extras: "
                "pip install -r requirements/asr.txt"
            ) from exc

        try:
            self._model = WhisperModel(
                str(self.model_dir),
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                local_files_only=self.local_files_only,
            )
        except Exception as exc:  # CTranslate2 raises several exception types
            hint = ""
            if self.device == "cuda":
                hint = (
                    " Check that the NVIDIA driver, CUDA libraries and the "
                    "faster-whisper/ctranslate2 build are compatible, or retry with --device cpu."
                )
                if "out of memory" in str(exc).lower():
                    hint += (
                        " GPU VRAM is exhausted: stop llama-server"
                        " (scripts/stop_llama_server.sh) or run with --skip-llm."
                    )
            raise BackendUnavailableError(
                f"failed to load faster-whisper model from {self.model_dir}: {exc}.{hint}"
            ) from exc

    def release(self) -> None:
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # -- inference -----------------------------------------------------------
    def transcribe(
        self,
        chunks: list[AudioChunk],
        language: str = "ru",
        initial_prompt: str | None = None,
    ) -> ASRResult:
        if self._model is None:
            self.load()
        assert self._model is not None

        result = ASRResult(backend=self.name, model=str(self.model_dir), language=language)
        result.words_available = self.word_timestamps
        raw_segments: list[tuple[AudioChunk, list[Segment]]] = []

        for chunk in chunks:
            segments = self._transcribe_chunk(chunk, language, initial_prompt)
            raw_segments.append((chunk, segments))

        result.segments = merge_chunk_segments(raw_segments)
        if self._detected_language:
            result.language = self._detected_language
            result.info["detected_language"] = self._detected_language
        if self.word_timestamps and result.segments:
            result.warnings.append(
                "word timestamps are provided by faster-whisper and are approximate"
            )
        return result

    def _transcribe_chunk(
        self, chunk: AudioChunk, language: str, initial_prompt: str | None
    ) -> list[Segment]:
        assert self._model is not None
        try:
            iterator, info = self._model.transcribe(
                str(chunk.path),
                language=language or None,
                task="transcribe",
                beam_size=self.beam_size,
                word_timestamps=self.word_timestamps,
                vad_filter=self.vad_filter,
                vad_parameters={"min_silence_duration_ms": self.vad_min_silence_ms},
                condition_on_previous_text=self.condition_on_previous_text,
                initial_prompt=initial_prompt or None,
                log_progress=False,
            )
            detected_language = getattr(info, "language", language)
            segments: list[Segment] = []
            for item in iterator:
                text = (item.text or "").strip()
                if not text:
                    continue
                words = [
                    Word(
                        start=chunk.start + float(word.start),
                        end=chunk.start + float(word.end),
                        word=word.word.strip(),
                        probability=float(word.probability) if word.probability is not None else None,
                    )
                    for word in (item.words or [])
                    if word.word and word.word.strip()
                ]
                flags: list[str] = []
                if item.avg_logprob is not None and item.avg_logprob < -1.0:
                    flags.append("low_asr_confidence")
                if item.no_speech_prob is not None and item.no_speech_prob > 0.6:
                    flags.append("possible_silence")
                if item.compression_ratio is not None and item.compression_ratio > 2.4:
                    flags.append("repetitive_text")
                segments.append(
                    Segment(
                        start=chunk.start + float(item.start),
                        end=chunk.start + float(item.end),
                        speaker_id=None,
                        raw_text=text,
                        words=words,
                        quality_flags=flags,
                    )
                )
            if detected_language:
                self._detected_language = detected_language
            return segments
        except Exception as exc:
            raise BackendUnavailableError(
                f"faster-whisper failed on chunk {chunk.index} ({chunk.path}): {exc}"
            ) from exc

    # -- helpers -------------------------------------------------------------
    def _validate_model_directory(self) -> None:
        if not self.model_dir.is_dir():
            raise ModelNotFoundError(
                f"ASR model directory not found: {self.model_dir}. "
                "Run ./scripts/download_models.sh (runtime never downloads weights)."
            )
        missing = [name for name in REQUIRED_FILES if not (self.model_dir / name).is_file()]
        if missing:
            raise ModelNotFoundError(
                f"ASR model directory {self.model_dir} is incomplete; missing: {', '.join(missing)}. "
                "Re-run ./scripts/download_models.sh."
            )


def merge_chunk_segments(
    chunks: list[tuple[AudioChunk, list[Segment]]],
) -> list[Segment]:
    """Merge per-chunk results, discarding duplicated overlap halves.

    Each chunk contributes only the part of its window outside the midpoint of
    its overlap with neighbours. A segment is kept when its midpoint falls in
    that region, which avoids cutting words while removing duplicates.
    """

    if len(chunks) == 1:
        segments = chunks[0][1]
        segments.sort(key=lambda s: (s.start, s.end))
        return segments

    merged: list[Segment] = []
    last_index = len(chunks) - 1
    for index, (chunk, segments) in enumerate(chunks):
        # Overlap is encoded in adjacent chunk windows; derive it from geometry.
        prev = chunks[index - 1][0] if index > 0 else None
        nxt = chunks[index + 1][0] if index < last_index else None
        left_bound = chunk.start
        right_bound = chunk.end
        if prev is not None:
            shared = prev.end - chunk.start
            if shared > 0:
                left_bound = chunk.start + shared / 2
        if nxt is not None:
            shared = chunk.end - nxt.start
            if shared > 0:
                right_bound = chunk.end - shared / 2
        for segment in segments:
            # A segment belongs to the chunk where it starts. Segments that
            # start inside the left half of an overlap are duplicates of the
            # previous chunk and are dropped here.
            if segment.start < left_bound:
                continue
            if nxt is not None and segment.start >= right_bound:
                continue
            merged.append(segment)

    merged.sort(key=lambda s: (s.start, s.end))
    # Guard against exact duplicates caused by zero-length overlap geometry.
    deduped: list[Segment] = []
    for segment in merged:
        if deduped and segment.raw_text == deduped[-1].raw_text and abs(segment.start - deduped[-1].start) < 0.05:
            continue
        deduped.append(segment)
    return deduped
