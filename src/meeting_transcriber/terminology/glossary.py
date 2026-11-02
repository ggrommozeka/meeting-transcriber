"""Glossary storage and profile loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..errors import ConfigError


@dataclass
class Term:
    canonical: str
    aliases: list[str] = field(default_factory=list)
    category: str | None = None
    project: str | None = None
    case_sensitive: bool = False

    @property
    def term_id(self) -> str:
        return self.canonical


@dataclass
class Glossary:
    terms: list[Term] = field(default_factory=list)
    profiles: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)

    def alias_entries(self) -> list[tuple[str, Term]]:
        entries: list[tuple[str, Term]] = []
        for term in self.terms:
            for alias in term.aliases:
                if alias.strip():
                    entries.append((alias, term))
            entries.append((term.canonical, term))
        return entries

    def extend(self, other: Glossary) -> None:
        self.terms.extend(other.terms)
        self.profiles.extend(p for p in other.profiles if p not in self.profiles)
        self.source_files.extend(other.source_files)


def load_glossary_file(path: str | Path) -> Glossary:
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise ConfigError(f"terminology file not found: {file_path}")
    try:
        payload = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid terminology YAML in {file_path}: {exc}") from exc
    if not isinstance(payload, dict) or "terms" not in payload:
        raise ConfigError(f"{file_path}: expected a mapping with a 'terms' list")
    terms_raw = payload["terms"]
    if not isinstance(terms_raw, list):
        raise ConfigError(f"{file_path}: 'terms' must be a list")

    terms: list[Term] = []
    for index, item in enumerate(terms_raw):
        if not isinstance(item, dict) or not item.get("canonical"):
            raise ConfigError(f"{file_path}: term #{index} must have a non-empty 'canonical'")
        aliases = item.get("aliases") or []
        if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
            raise ConfigError(f"{file_path}: term '{item['canonical']}' has invalid 'aliases'")
        terms.append(
            Term(
                canonical=str(item["canonical"]).strip(),
                aliases=[str(a).strip() for a in aliases if str(a).strip()],
                category=item.get("category"),
                project=item.get("project"),
                case_sensitive=bool(item.get("case_sensitive", False)),
            )
        )
    profile = payload.get("profile")
    profiles = payload.get("profiles") or ([profile] if profile else [])
    return Glossary(terms=terms, profiles=list(profiles), source_files=[str(file_path)])


def load_glossaries(
    profiles: list[str],
    terminology_dir: str | Path,
    extra_files: list[str] | None = None,
) -> Glossary:
    glossary = Glossary(profiles=list(profiles))
    base = Path(terminology_dir).expanduser()
    for profile in profiles:
        glossary.extend(load_glossary_file(base / f"{profile}.yaml"))
    for extra in extra_files or []:
        glossary.extend(load_glossary_file(extra))
    if not glossary.terms:
        raise ConfigError("no terminology terms were loaded")
    return glossary


def glossary_to_dict(glossary: Glossary) -> dict[str, Any]:
    return {
        "version": 1,
        "profiles": glossary.profiles,
        "terms": [
            {
                "canonical": term.canonical,
                "aliases": term.aliases,
                **({"category": term.category} if term.category else {}),
                **({"project": term.project} if term.project else {}),
                **({"case_sensitive": True} if term.case_sensitive else {}),
            }
            for term in glossary.terms
        ],
    }
