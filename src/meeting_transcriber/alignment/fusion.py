"""Fusion of ASR segments and diarization intervals.

The algorithm is purely temporal. It never uses text to guess speaker changes,
because that would silently "repair" diarization with language evidence.
Ambiguous assignments are left unlabelled or flagged instead of guessed.
"""

from __future__ import annotations

from dataclasses import replace

from ..types import DiarizationInterval, Segment, Speaker, Word


def _overlaps(
    start: float, end: float, intervals: list[DiarizationInterval]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for interval in intervals:
        left = max(start, interval.start)
        right = min(end, interval.end)
        if right > left:
            result[interval.speaker_id] = result.get(interval.speaker_id, 0.0) + (right - left)
    return result


def _choose_speaker(
    start: float, end: float, intervals: list[DiarizationInterval]
) -> tuple[str | None, dict[str, float], bool]:
    duration = max(end - start, 1e-9)
    overlap = _overlaps(start, end, intervals)
    if not overlap:
        return None, {}, False
    ranked = sorted(overlap.items(), key=lambda item: item[1], reverse=True)
    best_speaker, best_overlap = ranked[0]
    best_ratio = best_overlap / duration
    ambiguous = False
    if len(ranked) > 1:
        second_overlap = ranked[1][1]
        # Two speakers explain a comparable share of the segment: do not guess.
        if second_overlap >= 0.8 * best_overlap and best_ratio < 0.65:
            ambiguous = True
    if best_ratio < 0.1:
        return None, overlap, True
    return best_speaker, overlap, ambiguous


def assign_word_speakers(
    words: list[Word], intervals: list[DiarizationInterval]
) -> None:
    for word in words:
        speaker, overlap, ambiguous = _choose_speaker(word.start, word.end, intervals)
        ranked = sorted(overlap.values(), reverse=True)
        if ambiguous and len(ranked) > 1 and ranked[1] >= 0.8 * ranked[0]:
            word.speaker_id = None
        else:
            word.speaker_id = speaker


def align_segments(
    segments: list[Segment],
    intervals: list[DiarizationInterval],
) -> tuple[list[Segment], list[Speaker], list[str]]:
    """Assign speakers and split segments at speaker changes.

    Returns aligned segments, the speaker list and pipeline warnings.
    """

    warnings: list[str] = []
    if not intervals:
        for segment in segments:
            segment.speaker_id = None
            segment.quality_flags.append("no_diarization")
        return segments, [], ["diarization produced no intervals; speakers are unknown"]

    aligned: list[Segment] = []
    for segment in segments:
        speaker, overlap_map, ambiguous = _choose_speaker(segment.start, segment.end, intervals)
        segment.speaker_id = speaker
        segment.overlap_speakers = sorted(overlap_map)
        if ambiguous:
            segment.quality_flags.append("ambiguous_speaker_assignment")
        if speaker is None:
            segment.quality_flags.append("no_speaker_match")
            aligned.append(segment)
            continue

        assign_word_speakers(segment.words, intervals)
        if segment.words:
            groups = _group_words_by_speaker(segment.words, fallback=speaker)
            if len(groups) > 1:
                aligned.extend(_segments_from_groups(segment, groups))
                continue
        aligned.append(segment)

    speakers = _collect_speakers(aligned)
    if any(s.speaker_id is None for s in aligned):
        warnings.append(
            "some segments could not be attributed to a speaker; see quality_flags"
        )
    aligned.sort(key=lambda s: (s.start, s.end))
    return aligned, speakers, warnings


def _group_words_by_speaker(words: list[Word], fallback: str) -> list[list[Word]]:
    groups: list[list[Word]] = []
    for word in words:
        label = word.speaker_id or fallback
        normalized = replace(word, speaker_id=label)
        if groups and groups[-1][0].speaker_id == label:
            groups[-1].append(normalized)
        else:
            groups.append([normalized])
    return groups


def _segments_from_groups(segment: Segment, groups: list[list[Word]]) -> list[Segment]:
    result: list[Segment] = []
    for group in groups:
        if not group:
            continue
        text = " ".join(word.word for word in group if word.word).strip()
        if not text:
            continue
        speaker = group[0].speaker_id
        result.append(
            Segment(
                start=group[0].start,
                end=group[-1].end,
                speaker_id=speaker,
                raw_text=text,
                corrected_text="",
                words=list(group),
                quality_flags=list(segment.quality_flags) + ["split_at_speaker_change"],
                overlap_speakers=list(segment.overlap_speakers),
            )
        )
    return result


def _collect_speakers(segments: list[Segment]) -> list[Speaker]:
    speakers: list[Speaker] = []
    seen: set[str] = set()
    for segment in segments:
        if segment.speaker_id and segment.speaker_id not in seen:
            seen.add(segment.speaker_id)
            speakers.append(Speaker(speaker_id=segment.speaker_id))
    return speakers
