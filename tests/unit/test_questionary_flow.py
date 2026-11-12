from pathlib import Path

from meeting_transcriber.config import load_config
from meeting_transcriber.tui.common import (
    discover_media_files,
    validate_existing_dir,
    validate_media_file,
    validate_prompt_file,
)
from meeting_transcriber.tui.process_flow import (
    ProcessSettings,
    apply_settings,
    summary_rows,
)

ROOT = Path(__file__).resolve().parents[2]


def test_discover_media_files_filters_and_sorts(tmp_path):
    video = tmp_path / "meeting.mp4"
    audio = tmp_path / "call.wav"
    note = tmp_path / "notes.txt"
    for path in (video, audio, note):
        path.write_bytes(b"x")
    found = discover_media_files([tmp_path])
    assert set(found) == {video.resolve(), audio.resolve()}

    assert validate_media_file(str(video)) is True
    assert "не найден" in str(validate_media_file(str(tmp_path / "missing.mp4")))
    assert "Промпт должен" in str(validate_prompt_file(str(video)))
    assert validate_existing_dir(str(tmp_path)) is True


def test_apply_settings_uses_wizard_values(tmp_path):
    source = tmp_path / "meeting.mp4"
    source.write_bytes(b"x")
    output = tmp_path / "out"
    output.mkdir()
    settings = ProcessSettings(
        input_path=source,
        output_dir=output,
        visual=True,
        diarization=False,
        report=True,
        report_mode="status_meeting",
        custom_prompt=None,
        speakers=4,
    )
    cfg = load_config(ROOT / "config/default.yaml")
    apply_settings(cfg, settings)
    assert cfg.visual.enabled is True
    assert cfg.diarization.enabled is False
    assert cfg.diarization.num_speakers == 4
    assert cfg.llm.report_mode == "status_meeting"
    assert cfg.paths.output_dir == str(output)

    rows = dict(summary_rows(settings))
    assert "OCR" in rows["Функции"]
    assert "Статус-митинг" in rows["Функции"]
    assert "4" in rows["Спикеры"]
