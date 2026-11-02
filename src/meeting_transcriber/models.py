"""Local model manifest, verification and explicit download.

Runtime never calls this module. Models are fetched only by an explicit
``download-models`` / ``scripts/download_models.sh`` invocation on the host.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import ModelIntegrityError, ModelNotFoundError

LOCK_FILENAME = "manifest.lock.json"


@dataclass
class ModelFile:
    name: str
    url: str
    sha256: str | None
    size: int | None

    @property
    def display_name(self) -> str:
        return self.name


@dataclass
class ModelGroup:
    key: str
    name: str
    repo: str
    revision: str
    license: str
    license_url: str
    target_dir: str
    files: list[ModelFile]


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser()
    if not manifest_path.is_file():
        raise ModelNotFoundError(f"model manifest not found: {manifest_path}")
    try:
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ModelIntegrityError(f"invalid model manifest {manifest_path}: {exc}") from exc
    if not isinstance(data, dict) or "models" not in data:
        raise ModelIntegrityError(f"{manifest_path}: expected a 'models' mapping")
    return data


def iter_groups(manifest: dict[str, Any]) -> Iterable[ModelGroup]:
    for key, group in manifest["models"].items():
        files = []
        for item in group.get("files", []):
            files.append(
                ModelFile(
                    name=item["name"],
                    url=item["url"],
                    sha256=item.get("sha256"),
                    size=item.get("size"),
                )
            )
        yield ModelGroup(
            key=key,
            name=group["name"],
            repo=group["repo"],
            revision=group["revision"],
            license=group.get("license", "unknown"),
            license_url=group.get("license_url", ""),
            target_dir=group["target_dir"],
            files=files,
        )


def _sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _lock_path(models_dir: Path) -> Path:
    return models_dir / LOCK_FILENAME


def _read_lock(models_dir: Path) -> dict[str, Any]:
    path = _lock_path(models_dir)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_lock(models_dir: Path, data: dict[str, Any]) -> None:
    path = _lock_path(models_dir)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _human(size: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def _download_file(url: str, destination: Path, expected_size: int | None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "meeting-transcriber/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as handle:
            total = expected_size or _content_length(response)
            downloaded = 0
            last_report = 0.0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
                downloaded += len(block)
                if total and downloaded / total - last_report >= 0.05:
                    last_report = downloaded / total
                    print(
                        f"    {downloaded * 100 / total:5.1f}%  {_human(downloaded)} / {_human(total)}",
                        file=sys.stderr,
                    )
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        raise ModelIntegrityError(
            f"download failed for {url}: {exc}. Check network access and the model source."
        ) from exc
    if expected_size is not None and tmp.stat().st_size != expected_size:
        actual = tmp.stat().st_size
        tmp.unlink(missing_ok=True)
        raise ModelIntegrityError(
            f"size mismatch for {url}: expected {expected_size} bytes, got {actual}. "
            "The upstream file may have changed; update models/manifest.yaml."
        )
    tmp.replace(destination)


def _content_length(response: Any) -> int | None:
    value = response.headers.get("Content-Length")
    try:
        return int(value) if value else None
    except ValueError:
        return None


def download_models(
    manifest_path: str | Path,
    models_dir: str | Path,
    only: str = "all",
    force: bool = False,
) -> int:
    manifest = load_manifest(manifest_path)
    models_root = Path(models_dir).expanduser().resolve()
    models_root.mkdir(parents=True, exist_ok=True)
    lock = _read_lock(models_root)
    groups = [g for g in iter_groups(manifest) if only == "all" or g.key == only]
    if not groups:
        raise ModelNotFoundError(f"no model group named '{only}' in {manifest_path}")

    failures: list[str] = []
    for group in groups:
        print(f"\n== {group.key}: {group.name} ({group.license})", file=sys.stderr)
        target = models_root / group.target_dir
        target.mkdir(parents=True, exist_ok=True)
        for model_file in group.files:
            destination = target / model_file.name
            relative = str(destination.relative_to(models_root))
            recorded = lock.get(relative, {})
            if destination.is_file() and not force:
                actual = _sha256(destination)
                expected = model_file.sha256 or recorded.get("sha256")
                if expected and actual != expected:
                    print(f"  ! {relative}: checksum mismatch, re-downloading", file=sys.stderr)
                elif expected:
                    print(f"  = {relative}: already present and verified", file=sys.stderr)
                    lock[relative] = {"sha256": actual, "size": destination.stat().st_size}
                    continue
                else:
                    print(f"  = {relative}: already present (no recorded hash)", file=sys.stderr)
                    lock[relative] = {"sha256": actual, "size": destination.stat().st_size}
                    continue
            if group.repo == "local":
                failures.append(f"{relative}: missing local asset and no download URL")
                continue
            print(f"  + {relative} ({_human(model_file.size or 0)})", file=sys.stderr)
            try:
                _download_file(model_file.url, destination, model_file.size)
            except ModelIntegrityError as exc:
                failures.append(str(exc))
                continue
            actual = _sha256(destination)
            expected = model_file.sha256 or recorded.get("sha256")
            if expected and actual != expected:
                destination.unlink(missing_ok=True)
                failures.append(f"{relative}: SHA-256 mismatch (expected {expected}, got {actual})")
                continue
            lock[relative] = {"sha256": actual, "size": destination.stat().st_size}
            print(f"  ✓ {relative}: verified {actual[:16]}…", file=sys.stderr)

        license_note = (
            f"Model: {group.name}\n"
            f"Source: {group.repo} @ {group.revision}\n"
            f"License: {group.license}\n"
            f"License/card: {group.license_url}\n"
            f"Downloaded by meeting-transcriber's download_models.sh.\n"
            f"This file is informational; the upstream license governs usage.\n"
        )
        (target / "LICENSE.txt").write_text(license_note, encoding="utf-8")

    _write_lock(models_root, lock)
    if failures:
        print("\nDownload completed with errors:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"\nAll requested models are present under {models_root}", file=sys.stderr)
    return 0


def verify_models(
    manifest_path: str | Path,
    models_dir: str | Path,
    only: str = "all",
) -> int:
    manifest = load_manifest(manifest_path)
    models_root = Path(models_dir).expanduser().resolve()
    lock = _read_lock(models_root)
    groups = [g for g in iter_groups(manifest) if only == "all" or g.key == only]
    errors: list[str] = []
    checked = 0

    for group in groups:
        target = models_root / group.target_dir
        print(f"== {group.key}: {target}", file=sys.stderr)
        if not target.is_dir():
            errors.append(f"{group.key}: directory missing: {target}")
            continue
        for model_file in group.files:
            path = target / model_file.name
            relative = str(path.relative_to(models_root))
            if not path.is_file():
                errors.append(f"{relative}: missing")
                continue
            checked += 1
            actual_size = path.stat().st_size
            if model_file.size is not None and actual_size != model_file.size:
                errors.append(
                    f"{relative}: size mismatch (expected {model_file.size}, got {actual_size})"
                )
                continue
            actual_hash = _sha256(path)
            expected = model_file.sha256 or lock.get(relative, {}).get("sha256")
            if expected is None:
                print(f"  ? {relative}: no reference hash (run download-models to record one)", file=sys.stderr)
            elif actual_hash != expected:
                errors.append(f"{relative}: SHA-256 mismatch")
            else:
                print(f"  ✓ {relative}", file=sys.stderr)

    if errors:
        print("\nModel verification failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"\nVerified {checked} file(s). All model weights are local and intact.", file=sys.stderr)
    return 0


def list_models(manifest_path: str | Path, models_dir: str | Path) -> None:
    manifest = load_manifest(manifest_path)
    models_root = Path(models_dir).expanduser().resolve()
    for group in iter_groups(manifest):
        present = all((models_root / group.target_dir / f.name).is_file() for f in group.files)
        status = "present" if present else "missing"
        print(f"{group.key:22} {group.name:45} {group.revision[:12]:12} {status}")
        print(f"  license: {group.license}  source: {group.repo}")
