"""Optional extraction of terminology from a C++ code base.

The scanner is intentionally conservative: it reads source files only when the
user explicitly runs ``build-glossary`` with a path. It does not walk outside
that path and skips build artefacts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .glossary import Glossary, Term, glossary_to_dict

SKIP_DIRS = {".git", "build", "cmake-build-debug", "cmake-build-release", "third_party", "external", "node_modules"}
SOURCE_SUFFIXES = {".h", ".hpp", ".hh", ".hxx", ".c", ".cc", ".cpp", ".cxx", ".cmake", ".txt"}

PATTERNS = [
    (re.compile(r"\b(?:class|struct|union|enum(?:\s+class)?)\s+([A-Za-z_]\w*)"), "class"),
    (re.compile(r"\bnamespace\s+([A-Za-z_]\w*)"), "namespace"),
    (re.compile(r"\busing\s+([A-Za-z_]\w*)\s*="), "type"),
    (re.compile(r"\btypedef\b[^;]*?\b([A-Za-z_]\w*)\s*;"), "type"),
    (re.compile(r"\badd_(?:library|executable)\s*\(\s*([A-Za-z_][\w-]*)"), "cmake-target"),
    (re.compile(r"\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:const)?\s*\{"), "function"),
]


@dataclass
class ScanResult:
    files_scanned: int
    terms: list[Term]


def scan_cpp_repository(root: str | Path, max_files: int = 5000) -> ScanResult:
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise ValueError(f"project path is not a directory: {base}")

    found: dict[str, Term] = {}
    scanned = 0
    for path in sorted(base.rglob("*")):
        if scanned >= max_files:
            break
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(base).parts[:-1]):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        scanned += 1
        for pattern, category in PATTERNS:
            for match in pattern.finditer(text):
                name = match.group(1)
                if len(name) < 3 or name in found:
                    continue
                found[name] = Term(canonical=name, aliases=[], category=category, project=base.name)
    terms = sorted(found.values(), key=lambda t: (t.category or "", t.canonical.casefold()))
    return ScanResult(files_scanned=scanned, terms=terms)


def write_glossary(terms: list[Term], output_path: str | Path, profile: str | None = None) -> Path:
    import yaml

    glossary = Glossary(terms=terms, profiles=[profile] if profile else [])
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(glossary_to_dict(glossary), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path
