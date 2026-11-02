from .codebase_import import ScanResult, scan_cpp_repository, write_glossary
from .correction import TerminologyCorrector
from .glossary import Glossary, Term, glossary_to_dict, load_glossaries, load_glossary_file
from .matcher import Match, apply_matches, find_matches, normalize_token, phonetic_key, tokenize

__all__ = [
    "Glossary",
    "Match",
    "ScanResult",
    "Term",
    "TerminologyCorrector",
    "apply_matches",
    "find_matches",
    "glossary_to_dict",
    "load_glossaries",
    "load_glossary_file",
    "normalize_token",
    "phonetic_key",
    "scan_cpp_repository",
    "tokenize",
    "write_glossary",
]
