"""Local OCR backend.

Only Tesseract is used: it is CPU-only, works offline with local
``*.traineddata`` files and does not compete with ASR/diarization for VRAM.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from PIL import Image

from ..config import VisualConfig


class TesseractOCR:
    def __init__(self, cfg: VisualConfig) -> None:
        self.cfg = cfg
        self.binary = shutil.which(cfg.tesseract)
        self.languages = "+".join(lang for lang in cfg.languages if lang) or "eng"
        self.env = os.environ.copy()
        if cfg.tessdata_dir:
            self.env["TESSDATA_PREFIX"] = str(Path(cfg.tessdata_dir).expanduser())

    @property
    def available(self) -> bool:
        return bool(self.binary)

    def recognize(self, image: Path) -> tuple[str, str | None]:
        """Return ``(text, warning)``; OCR failures are non-fatal."""

        if not self.binary:
            return "", (
                "tesseract executable not found; install tesseract-ocr with rus and eng "
                "language data or disable visual.enabled"
            )
        ocr_image = self._cropped(image)
        command = [
            self.binary,
            str(ocr_image),
            "stdout",
            "-l",
            self.languages,
            "--psm",
            "6",
            "-c",
            "preserve_interword_spaces=1",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.cfg.ocr_timeout_seconds,
                env=self.env,
            )
        except subprocess.TimeoutExpired:
            return "", f"OCR timed out for {image.name}"
        finally:
            if ocr_image != image:
                ocr_image.unlink(missing_ok=True)
        if completed.returncode != 0:
            return "", (
                f"OCR failed for {image.name}: "
                f"{completed.stderr.strip() or 'unknown error'}"
            )
        return completed.stdout or "", None

    def _cropped(self, image: Path) -> Path:
        left, top, right, bottom = self.cfg.crop
        if (left, top, right, bottom) == (0.0, 0.0, 1.0, 1.0):
            return image
        target = image.with_name(f"{image.stem}_crop{image.suffix}")
        with Image.open(image) as source:
            width, height = source.size
            box = (
                int(width * left),
                int(height * top),
                int(width * right),
                int(height * bottom),
            )
            source.crop(box).save(target, quality=95)
        return target


__all__ = ["TesseractOCR"]
