from meeting_transcriber.alignment.fusion import align_segments
from meeting_transcriber.types import DiarizationInterval, Segment, Word


def test_segment_is_assigned_to_dominant_speaker():
    segments = [Segment(start=0.0, end=5.0, speaker_id=None, raw_text="hello world")]
    intervals = [
        DiarizationInterval(start=0.0, end=4.0, speaker_id="speaker_00"),
        DiarizationInterval(start=4.0, end=8.0, speaker_id="speaker_01"),
    ]
    aligned, speakers, warnings = align_segments(segments, intervals)
    assert aligned[0].speaker_id == "speaker_00"
    assert aligned[0].overlap_speakers == ["speaker_00", "speaker_01"]
    assert [s.speaker_id for s in speakers] == ["speaker_00"]
    assert warnings == []


def test_segments_split_at_word_level_speaker_change():
    words = [
        Word(start=0.0, end=1.0, word="привет", probability=0.9),
        Word(start=2.0, end=3.0, word="коллеги", probability=0.9),
        Word(start=4.0, end=5.0, word="давайте", probability=0.9),
    ]
    segments = [Segment(start=0.0, end=5.0, speaker_id=None, raw_text="привет коллеги давайте", words=words)]
    intervals = [
        DiarizationInterval(start=0.0, end=3.5, speaker_id="speaker_00"),
        DiarizationInterval(start=3.5, end=6.0, speaker_id="speaker_01"),
    ]
    aligned, speakers, _ = align_segments(segments, intervals)
    assert [s.raw_text for s in aligned] == ["привет коллеги", "давайте"]
    assert [s.speaker_id for s in aligned] == ["speaker_00", "speaker_01"]
    assert len(speakers) == 2


def test_ambiguous_overlap_is_flagged_not_guessed():
    segments = [Segment(start=0.0, end=4.0, speaker_id=None, raw_text="overlap")]
    intervals = [
        DiarizationInterval(start=0.0, end=2.0, speaker_id="speaker_00"),
        DiarizationInterval(start=2.0, end=4.0, speaker_id="speaker_01"),
    ]
    aligned, _, _ = align_segments(segments, intervals)
    assert aligned[0].speaker_id in {"speaker_00", "speaker_01"}
    assert "ambiguous_speaker_assignment" in aligned[0].quality_flags


def test_no_diarization_keeps_segments_unlabelled():
    segments = [Segment(start=0.0, end=1.0, speaker_id=None, raw_text="text")]
    aligned, speakers, warnings = align_segments(segments, [])
    assert aligned[0].speaker_id is None
    assert "no_diarization" in aligned[0].quality_flags
    assert speakers == []
    assert warnings
