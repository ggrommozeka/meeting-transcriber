
from meeting_transcriber.config import DiarizationConfig
from meeting_transcriber.diarization.nemo_backend import (
    build_diarizer_config,
    normalize_speaker_ids,
    parse_rttm,
    write_main_manifest,
    write_vad_manifest,
)
from meeting_transcriber.types import DiarizationInterval


def test_rttm_is_parsed_and_sorted(tmp_path):
    rttm = tmp_path / "meeting.rttm"
    rttm.write_text(
        "SPEAKER meeting 1 5.000 1.500 <NA> <NA> speaker_1 <NA> <NA>\n"
        "SPEAKER meeting 1 0.000 2.000 <NA> <NA> speaker_0 <NA> <NA>\n"
        "SPKR-INFO meeting 1 <NA> <NA> <NA> unknown speaker_0 <NA> <NA>\n",
        encoding="utf-8",
    )
    intervals = parse_rttm(rttm)
    assert [(i.start, i.end, i.speaker_id) for i in intervals] == [
        (0.0, 2.0, "speaker_0"),
        (5.0, 6.5, "speaker_1"),
    ]


def test_speaker_ids_are_normalized_by_first_appearance():
    intervals = [
        DiarizationInterval(10.0, 11.0, "7"),
        DiarizationInterval(1.0, 2.0, "3"),
        DiarizationInterval(20.0, 21.0, "7"),
    ]
    normalized = normalize_speaker_ids(intervals)
    assert [i.speaker_id for i in normalized] == ["speaker_00", "speaker_01", "speaker_00"]


def test_manifests_have_required_fields(tmp_path):
    wav = tmp_path / "audio_16k_mono.wav"
    wav.write_bytes(b"")
    vad = tmp_path / "vad.json"
    write_vad_manifest(vad, wav, "meeting", [(0.0, 1.0), (2.5, 3.0)])
    lines = [line for line in vad.read_text().splitlines() if line]
    assert len(lines) == 2
    main = tmp_path / "main.json"
    write_main_manifest(main, wav, "meeting", 12)
    import json

    entry = json.loads(main.read_text())
    assert entry["uniq_id"] == "meeting"
    assert entry["num_speakers"] == 12


def test_diarizer_config_supports_more_than_four_speakers(tmp_path):
    cfg = DiarizationConfig(max_speakers=20)
    built = build_diarizer_config(
        cfg,
        manifest_path=tmp_path / "main.json",
        vad_manifest_path=tmp_path / "vad.json",
        speaker_model_path=tmp_path / "titanet.nemo",
        out_dir=tmp_path / "out",
    )
    clustering = built["diarizer"]["clustering"]["parameters"]
    assert clustering["max_num_speakers"] == 20
    assert clustering["oracle_num_speakers"] is False
    assert built["verbose"] is False
    assert built["diarizer"]["vad"]["model_path"] is None
    assert built["diarizer"]["vad"]["external_vad_manifest"].endswith("vad.json")


def test_explicit_speaker_count_uses_oracle(tmp_path):
    cfg = DiarizationConfig(max_speakers=20, num_speakers=12)
    built = build_diarizer_config(
        cfg,
        manifest_path=tmp_path / "main.json",
        vad_manifest_path=tmp_path / "vad.json",
        speaker_model_path=tmp_path / "titanet.nemo",
        out_dir=tmp_path / "out",
    )
    assert built["diarizer"]["clustering"]["parameters"]["oracle_num_speakers"] is True


def test_single_scale_config_for_long_audio(tmp_path):
    cfg = DiarizationConfig(
        window_lengths=1.5,
        shift_lengths=0.75,
        multiscale_weights=[1],
    )
    built = build_diarizer_config(
        cfg,
        manifest_path=tmp_path / "main.json",
        vad_manifest_path=tmp_path / "vad.json",
        speaker_model_path=tmp_path / "titanet.nemo",
        out_dir=tmp_path / "out",
    )
    params = built["diarizer"]["speaker_embeddings"]["parameters"]
    assert params["window_length_in_sec"] == 1.5
    assert params["shift_length_in_sec"] == 0.75
    assert params["multiscale_weights"] is None
