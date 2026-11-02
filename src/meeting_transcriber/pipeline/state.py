"""Pipeline run state used for safe resume."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

STATE_VERSION = 1


def config_fingerprint(config_dict: dict[str, Any]) -> str:
    payload = json.dumps(config_dict, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_state(work_dir: Path) -> dict[str, Any]:
    path = work_dir / "state.json"
    if not path.is_file():
        return {"version": STATE_VERSION, "stages": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"version": STATE_VERSION, "stages": {}}
    if not isinstance(data, dict) or data.get("version") != STATE_VERSION:
        return {"version": STATE_VERSION, "stages": {}}
    data.setdefault("stages", {})
    return data


def save_state(work_dir: Path, state: dict[str, Any]) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "state.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def stage_done(
    state: dict[str, Any],
    stage: str,
    artifact: Path,
    fingerprint: str | None = None,
) -> bool:
    entry = state.get("stages", {}).get(stage)
    if not (entry and entry.get("completed") and artifact.is_file()):
        return False
    if fingerprint is not None and entry.get("fingerprint") != fingerprint:
        return False
    return True


def mark_done(
    state: dict[str, Any],
    stage: str,
    artifact: Path,
    fingerprint: str | None = None,
    **extra: Any,
) -> None:
    state.setdefault("stages", {})[stage] = {
        "completed": True,
        "artifact": str(artifact),
        "fingerprint": fingerprint,
        **extra,
    }


def reset_state(work_dir: Path) -> None:
    path = work_dir / "state.json"
    path.unlink(missing_ok=True)
