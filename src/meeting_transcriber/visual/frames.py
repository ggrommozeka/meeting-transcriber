"""Frame sampling and screen-state grouping.

Two FFmpeg passes provide candidate frames:

* a periodic pass (``fps=1/interval``) that catches slowly changing slides;
* a scene-change pass (``select='gt(scene,threshold)'``) that catches short code
  flashes between periodic samples.

Candidates are grouped by a 64-bit difference hash. Consecutive frames with a
small Hamming distance form one stable screen state; the sharpest frame of the
state is kept for OCR.
"""

from __future__ import annotations

import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree

from PIL import Image, ImageFilter, ImageStat

from ..config import VisualConfig
from ..errors import AudioError

_PTS_RE = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")


@dataclass
class FrameSample:
    timestamp: float
    path: Path
    fingerprint: int = 0
    sharpness: float = 0.0


@dataclass
class FrameGroup:
    start: float
    end: float
    sample: FrameSample


def _resize(image: Image.Image, width: int) -> Image.Image:
    if image.width <= width:
        return image
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.BILINEAR)


def dhash(path: Path, size: int = 8) -> int:
    """Return a 64-bit difference hash for an image."""

    with Image.open(path) as image:
        gray = image.convert("L").resize(
            (size + 1, size), Image.Resampling.BILINEAR
        )
        pixels = gray.tobytes()
        width = size + 1
    value = 0
    for row in range(size):
        offset = row * width
        for column in range(size):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


def sharpness(path: Path) -> float:
    """Edge-energy proxy used to pick the least blurry frame of a group."""

    with Image.open(path) as image:
        gray = _resize(image.convert("L"), 640).filter(ImageFilter.FIND_EDGES)
        return float(ImageStat.Stat(gray).var[0])


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _periodic_frames(
    ffmpeg: str,
    source: Path,
    directory: Path,
    interval: float,
) -> list[FrameSample]:
    pattern = directory / "periodic_%06d.jpg"
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vf",
        f"fps=1/{interval},scale='min(1920,iw)':-2",
        "-vsync",
        "vfr",
        "-q:v",
        "2",
        str(pattern),
    ]
    completed = _run(command)
    if completed.returncode != 0:
        raise AudioError(
            "FFmpeg failed to extract periodic frames: "
            f"{completed.stderr.strip() or 'unknown error'}"
        )
    samples: list[FrameSample] = []
    for path in sorted(directory.glob("periodic_*.jpg")):
        index = int(path.stem.split("_")[-1]) - 1
        samples.append(FrameSample(timestamp=index * interval, path=path))
    return samples


def _scene_frames(
    ffmpeg: str,
    source: Path,
    directory: Path,
    threshold: float,
) -> list[FrameSample]:
    pattern = directory / "scene_%06d.jpg"
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "info",
        "-y",
        "-i",
        str(source),
        "-vf",
        f"select='gt(scene,{threshold})',showinfo,scale='min(1920,iw)':-2",
        "-vsync",
        "vfr",
        "-q:v",
        "2",
        str(pattern),
    ]
    completed = _run(command)
    if completed.returncode != 0:
        # Scene detection is a bonus: never fail the whole visual stage because
        # this optional pass did not work.
        return []
    timestamps = [float(match) for match in _PTS_RE.findall(completed.stderr)]
    paths = sorted(directory.glob("scene_*.jpg"))
    samples: list[FrameSample] = []
    for timestamp, path in zip(timestamps, paths, strict=False):
        samples.append(FrameSample(timestamp=timestamp, path=path))
    return samples


def extract_samples(
    ffmpeg: str,
    source: str | Path,
    work_dir: str | Path,
    cfg: VisualConfig,
) -> list[FrameSample]:
    """Extract and merge periodic and scene-change frames."""

    src = Path(source).expanduser().resolve()
    directory = Path(work_dir).expanduser().resolve() / "visual_frames"
    if directory.is_dir():
        rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)

    samples = _periodic_frames(ffmpeg, src, directory, cfg.frame_interval_seconds)
    samples += _scene_frames(ffmpeg, src, directory, cfg.scene_threshold)
    if not samples:
        return []

    samples.sort(key=lambda item: item.timestamp)
    merged: list[FrameSample] = []
    for sample in samples:
        if merged and abs(sample.timestamp - merged[-1].timestamp) < 0.5:
            continue
        merged.append(sample)

    if len(merged) > cfg.max_frames:
        step = math.ceil(len(merged) / cfg.max_frames)
        merged = merged[::step][: cfg.max_frames]
    return merged


def group_samples(
    samples: list[FrameSample],
    duration: float,
    dedupe_distance: int,
) -> list[FrameGroup]:
    """Group consecutive visually similar frames into stable screen states."""

    if not samples:
        return []
    for sample in samples:
        sample.fingerprint = dhash(sample.path)
        sample.sharpness = sharpness(sample.path)

    groups: list[FrameGroup] = []
    current = [samples[0]]
    for sample in samples[1:]:
        distance = (sample.fingerprint ^ current[-1].fingerprint).bit_count()
        if distance <= dedupe_distance:
            current.append(sample)
        else:
            groups.append(_finish_group(current, duration, next_start=sample.timestamp))
            current = [sample]
    groups.append(_finish_group(current, duration, next_start=duration))
    return groups


def _finish_group(
    samples: list[FrameSample], duration: float, next_start: float
) -> FrameGroup:
    best = max(samples, key=lambda item: item.sharpness)
    start = samples[0].timestamp
    end = min(next_start, duration) if next_start > start else duration
    return FrameGroup(start=start, end=end, sample=best)


def save_code_frame(group: FrameGroup, output_dir: Path, code_id: str) -> str:
    """Copy the selected frame into the persistent frames directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    name = f"{code_id}.jpg"
    target = output_dir / name
    target.write_bytes(group.sample.path.read_bytes())
    return name


__all__ = [
    "FrameGroup",
    "FrameSample",
    "dhash",
    "extract_samples",
    "group_samples",
    "save_code_frame",
    "sharpness",
]
