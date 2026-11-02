import hashlib
from pathlib import Path

import yaml

from meeting_transcriber.models import (
    download_models,
    iter_groups,
    load_manifest,
    verify_models,
)

ROOT = Path(__file__).resolve().parents[2]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(tmp_path: Path, payload: bytes) -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "models": {
                    "asr": {
                        "name": "tiny",
                        "repo": "local",
                        "revision": "0",
                        "license": "MIT",
                        "license_url": "",
                        "target_dir": "tiny",
                        "files": [
                            {
                                "name": "model.bin",
                                "url": "",
                                "sha256": _sha(payload),
                                "size": len(payload),
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_repository_manifest_is_well_formed():
    manifest = load_manifest(ROOT / "models/manifest.yaml")
    groups = list(iter_groups(manifest))
    assert {g.key for g in groups} == {"asr", "diarization", "llm"}
    for group in groups:
        assert group.revision
        assert group.files
        for model_file in group.files:
            assert model_file.url.startswith("https://")
            assert model_file.sha256 and len(model_file.sha256) == 64


def test_verify_detects_corruption(tmp_path):
    payload = b"model-bytes"
    manifest = _manifest(tmp_path, payload)
    models_dir = tmp_path / "models"
    target = models_dir / "tiny"
    target.mkdir(parents=True)
    (target / "model.bin").write_bytes(b"corrupted")
    assert verify_models(manifest, models_dir) == 1

    (target / "model.bin").write_bytes(payload)
    assert verify_models(manifest, models_dir) == 0


def test_verify_detects_missing_file(tmp_path):
    manifest = _manifest(tmp_path, b"x")
    assert verify_models(manifest, tmp_path / "models") == 1


def test_download_skips_local_assets_without_url(tmp_path):
    manifest = _manifest(tmp_path, b"x")
    models_dir = tmp_path / "models"
    assert download_models(manifest, models_dir) == 1  # no URL and file absent
