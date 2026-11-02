from pathlib import Path

import pytest
import yaml

from meeting_transcriber.config import TerminologyConfig
from meeting_transcriber.errors import ConfigError
from meeting_transcriber.terminology.correction import TerminologyCorrector
from meeting_transcriber.terminology.glossary import load_glossaries, load_glossary_file
from meeting_transcriber.terminology.matcher import find_matches, phonetic_key, tokenize
from meeting_transcriber.types import Segment, Word

ROOT = Path(__file__).resolve().parents[2]
TERMS = ROOT / "config/terminology"


def _corrector(**overrides):
    cfg = TerminologyConfig(profiles=["general-it", "example-project"], **overrides)
    return TerminologyCorrector.from_config(cfg, TERMS)


def test_glossary_loads_profiles():
    glossary = load_glossaries(["general-it", "example-project"], TERMS)
    canonicals = {term.canonical for term in glossary.terms}
    assert "CMake" in canonicals
    assert "vtkChartXY" in canonicals
    assert len(glossary.profiles) == 2


def test_unknown_glossary_file_is_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_glossary_file(tmp_path / "missing.yaml")


def test_exact_alias_correction_is_auditable():
    corrector = _corrector()
    segment = Segment(
        start=0.0,
        end=4.0,
        speaker_id="speaker_00",
        raw_text="Нужно проверить витэка чарт икс уай.",
        words=[
            Word(0.0, 1.0, "Нужно"),
            Word(1.0, 2.0, "проверить"),
            Word(2.0, 3.0, "витэка"),
            Word(3.0, 4.0, "чарт"),
        ],
    )
    corrector.correct_segments([segment])
    assert "vtkChartXY" in segment.corrected_text
    assert "витэка" not in segment.corrected_text
    assert segment.corrections
    correction = segment.corrections[0]
    assert correction.target == "vtkChartXY"
    assert correction.source == "витэка чарт икс уай"
    assert correction.confidence >= 0.86


def test_already_canonical_is_not_changed():
    corrector = _corrector()
    segment = Segment(start=0, end=1, speaker_id=None, raw_text="Открываю CMake и VTK.")
    corrector.correct_segments([segment])
    assert segment.corrected_text == "Открываю CMake и VTK."
    assert segment.corrections == []


def test_apply_corrections_can_be_disabled():
    cfg = TerminologyConfig(profiles=["general-it"], apply_corrections=False)
    corrector = TerminologyCorrector.from_config(cfg, TERMS)
    segment = Segment(start=0, end=1, speaker_id=None, raw_text="Открываю ви ти кей.")
    corrector.correct_segments([segment])
    assert segment.corrected_text == "Открываю ви ти кей."
    assert segment.corrections[0].target == "VTK"


def test_common_russian_words_are_not_replaced():
    corrector = _corrector()
    segment = Segment(
        start=0,
        end=1,
        speaker_id=None,
        raw_text="Куда мы идём? Надо удержаться и не упасть.",
    )
    corrector.correct_segments([segment])
    assert segment.corrected_text == segment.raw_text
    assert segment.corrections == []


def test_low_threshold_fuzzy_matching():
    cfg = TerminologyConfig(profiles=["general-it"], threshold=0.7, ambiguous_margin=0.2)
    glossary = load_glossaries(["general-it"], TERMS)
    matches, _ = find_matches("си мейк и гит", glossary, cfg)
    targets = {m.term.canonical for m in matches}
    assert "CMake" in targets or "Git" in targets


def test_tokenizer_keeps_identifiers():
    tokens = tokenize("Вызови vtkChartXY(1) и dirty-флаг")
    assert [t.normalized for t in tokens] == ["вызови", "vtkchartxy", "1", "и", "dirty", "флаг"]


def test_phonetic_key_matches_transliteration_variants():
    assert phonetic_key("витэка") == phonetic_key("viteka")


def test_glossary_file_schema(tmp_path):
    path = tmp_path / "terms.yaml"
    path.write_text(
        yaml.safe_dump(
            {"version": 1, "terms": [{"canonical": "API", "aliases": ["апи"], "category": "concept"}]},
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    glossary = load_glossary_file(path)
    assert glossary.terms[0].aliases == ["апи"]
