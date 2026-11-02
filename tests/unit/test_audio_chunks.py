from pathlib import Path

import pytest

from meeting_transcriber.asr.whisper_backend import merge_chunk_segments
from meeting_transcriber.audio.preprocess import (
    AudioChunk,
    build_convert_to_wav_command,
    build_extract_audio_command,
    plan_chunks,
)
from meeting_transcriber.types import Segment


def test_single_chunk_when_recording_is_short():
    assert plan_chunks(100.0, 900.0, 15.0) == [(0.0, 100.0)]


def test_chunks_cover_the_whole_recording():
    duration = 3600.0
    windows = plan_chunks(duration, 900.0, 15.0)
    assert windows[0][0] == 0.0
    assert windows[-1][1] == duration
    for (start, end), (next_start, _) in zip(windows, windows[1:], strict=False):
        assert next_start < end  # overlap
        assert next_start > start
    for start, end in windows:
        assert end - start <= 900.0 + 1e-6


def test_overlap_is_validated():
    with pytest.raises(ValueError):
        plan_chunks(10.0, 10.0, 10.0)


def test_merge_removes_duplicate_overlap_segments():
    chunks = [
        (
            AudioChunk(0, 0.0, 100.0, Path("a.wav")),
            [
                Segment(start=10.0, end=20.0, speaker_id=None, raw_text="first"),
                Segment(start=90.0, end=99.0, speaker_id=None, raw_text="boundary"),
            ],
        ),
        (
            AudioChunk(1, 85.0, 185.0, Path("b.wav")),
            [
                Segment(start=86.0, end=95.0, speaker_id=None, raw_text="boundary"),
                Segment(start=120.0, end=130.0, speaker_id=None, raw_text="second"),
            ],
        ),
    ]
    merged = merge_chunk_segments(chunks)
    texts = [segment.raw_text for segment in merged]
    assert texts == ["first", "boundary", "second"]


def test_merge_sorts_single_chunk():
    chunks = [
        (
            AudioChunk(0, 0.0, 10.0, Path("a.wav")),
            [
                Segment(start=5.0, end=6.0, speaker_id=None, raw_text="b"),
                Segment(start=1.0, end=2.0, speaker_id=None, raw_text="a"),
            ],
        )
    ]
    merged = merge_chunk_segments(chunks)
    assert [s.raw_text for s in merged] == ["a", "b"]


def test_extract_audio_command_selects_video_audio_stream():
    command = build_extract_audio_command(
        "ffmpeg", "meeting.mp4", "audio_original.mka", audio_stream=2, codec="copy"
    )
    assert command[command.index("-map") + 1] == "0:a:2"
    assert command[command.index("-c:a") + 1] == "copy"
    assert "-vn" in command
    assert command[-1] == "audio_original.mka"


def test_extract_audio_command_sets_explicit_container_format():
    command = build_extract_audio_command(
        "ffmpeg",
        "meeting.webm",
        "audio_original.part.mka",
        codec="copy",
        container_format="matroska",
    )
    assert command[command.index("-f") + 1] == "matroska"
    assert command[-1] == "audio_original.part.mka"


def test_convert_command_targets_mono_16k_pcm():
    command = build_convert_to_wav_command("ffmpeg", "audio_original.mka", "audio.wav", 0)
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "16000"
    assert command[command.index("-c:a") + 1] == "pcm_s16le"
    assert command[command.index("-map") + 1] == "0:a:0"
