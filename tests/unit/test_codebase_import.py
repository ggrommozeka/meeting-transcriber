from pathlib import Path

from meeting_transcriber.terminology.codebase_import import (
    scan_cpp_repository,
    write_glossary,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample.cpp"


def test_cpp_scan_finds_types_and_cmake_targets(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "widget.h").write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    (project / "CMakeLists.txt").write_text(
        "add_library(granum_core STATIC widget.cpp)\n", encoding="utf-8"
    )
    # Build directories must be ignored.
    build = project / "build"
    build.mkdir()
    (build / "generated.h").write_text("class Ignored {};\n", encoding="utf-8")

    result = scan_cpp_repository(project)
    canonicals = {term.canonical for term in result.terms}
    assert "vtkChartXY" in canonicals
    assert "GranumNamespace" in canonicals
    assert "granum_core" in canonicals
    assert "Ignored" not in canonicals
    assert result.files_scanned == 2


def test_write_glossary_is_loadable(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "a.cpp").write_text("struct FooBar {};\n", encoding="utf-8")
    result = scan_cpp_repository(project)
    output = write_glossary(result.terms, tmp_path / "generated.yaml", profile="generated")
    assert output.is_file()
    assert "FooBar" in output.read_text(encoding="utf-8")
