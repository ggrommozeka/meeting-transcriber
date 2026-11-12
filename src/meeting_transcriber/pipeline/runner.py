"""End-to-end pipeline orchestration."""

from __future__ import annotations

import gc
import json
import platform
import re
import shutil
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import __version__
from ..alignment.fusion import align_segments
from ..asr.base import ASRResult, Transcriber
from ..asr.whisper_backend import WhisperBackend
from ..audio.metadata import file_sha256
from ..audio.preprocess import PreparedAudio, plan_chunks, preprocess_audio
from ..config import AppConfig
from ..diarization.base import DiarizationResult, Diarizer
from ..diarization.nemo_backend import NemoDiarizer
from ..errors import ConfigError, PipelineError, TranscriberError
from ..output.json_writer import write_json
from ..output.markdown_writer import write_markdown
from ..output.srt_writer import write_srt
from ..postprocessing.meeting_report import generate_meeting_report, render_report_markdown
from ..speakers import apply_speaker_names, load_speaker_map
from ..terminology.correction import TerminologyCorrector
from ..types import PipelineRunInfo, Transcript
from ..visual import VisualResult, extract_visual, load_visual_result
from ..visual.types import CodeCard
from . import persistence
from . import state as state_mod


@dataclass
class RunResult:
    transcript: Transcript
    outputs: dict[str, Path] = field(default_factory=dict)
    report: dict[str, Any] | None = None
    work_dir: Path | None = None
    visual: VisualResult | None = None


def sanitize_meeting_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return cleaned or "meeting"


class Pipeline:
    def __init__(
        self,
        cfg: AppConfig,
        force: bool = False,
        skip_diarization: bool = False,
        skip_llm: bool = False,
        num_speakers: int | None = None,
        max_speakers: int | None = None,
        device: str | None = None,
        profile: str | None = None,
        output_dir: str | None = None,
        speaker_map: str | None = None,
        progress: Any | None = None,
        before_report: Callable[[], None] | None = None,
        after_report: Callable[[], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.force = force
        self.progress = progress
        self.before_report = before_report
        self.after_report = after_report
        self.skip_diarization = skip_diarization
        self.skip_llm = skip_llm
        self.speaker_map_path = Path(speaker_map).expanduser() if speaker_map else None
        self.speaker_map = (
            load_speaker_map(self.speaker_map_path) if self.speaker_map_path else {}
        )
        if num_speakers is not None:
            cfg.diarization.num_speakers = num_speakers
        if max_speakers is not None:
            cfg.diarization.max_speakers = max_speakers
        if device is not None:
            cfg.asr.device = device
            cfg.diarization.device = device
        if profile is not None:
            cfg.terminology.profiles = [profile]
        if output_dir is not None:
            cfg.paths.output_dir = output_dir

    # -- public --------------------------------------------------------------
    def run(self, input_path: str | Path) -> RunResult:
        source = Path(input_path).expanduser().resolve()
        if not source.is_file():
            raise PipelineError(f"audio file not found: {source}")
        meeting_id = sanitize_meeting_id(self.cfg.meeting_id or source.stem)
        work_dir = self.cfg.resolve(self.cfg.paths.work_dir) / meeting_id
        work_dir.mkdir(parents=True, exist_ok=True)

        stat = source.stat()
        source_identity = f"{source}:{stat.st_size}:{stat.st_mtime_ns}"
        fingerprint = state_mod.config_fingerprint(
            {
                "config": self.cfg.to_dict(),
                "source": source_identity,
                "options": {
                    "skip_diarization": self.skip_diarization,
                    "skip_llm": self.skip_llm,
                },
            }
        )
        state = state_mod.load_state(work_dir)
        if self.force:
            state = {"version": state_mod.STATE_VERSION, "stages": {}}
        state["config_fingerprint"] = fingerprint

        fingerprints = self._stage_fingerprints(source_identity)

        run = PipelineRunInfo(
            started_at=_now(),
            tool_version=__version__,
            device=self.cfg.asr.device,
            python_version=platform.python_version(),
            library_versions=_library_versions(),
        )
        warnings: list[str] = []
        timings: dict[str, float] = {}

        # 1. audio ---------------------------------------------------------
        started = time.perf_counter()
        with self._stage_progress("preprocess", "Извлечение аудио"):
            prepared = self._stage_preprocess(
                source, work_dir, state, fingerprints["preprocess"]
            )
        timings["preprocess"] = round(time.perf_counter() - started, 3)

        # 2. visual screen analysis (optional, CPU) ----------------------
        started = time.perf_counter()
        if self.cfg.visual.enabled and prepared.metadata.has_video:
            with self._stage_progress("visual", "Анализ кадров видео"):
                visual = self._stage_visual(
                    source, prepared, work_dir, state, fingerprints["visual"]
                )
        else:
            visual = self._stage_visual(
                source, prepared, work_dir, state, fingerprints["visual"]
            )
            if self.progress is not None:
                self.progress.skip_stage("visual", "Анализ кадров видео")
        timings["visual"] = round(time.perf_counter() - started, 3)
        warnings.extend(visual.warnings)

        # 3. ASR -----------------------------------------------------------
        started = time.perf_counter()
        _reset_peak()
        with self._stage_progress("asr", "Распознавание речи"):
            asr_result = self._stage_asr(
                prepared, work_dir, state, fingerprints["asr"]
            )
        timings["asr"] = round(time.perf_counter() - started, 3)
        peak_asr = _peak_mb()
        warnings.extend(asr_result.warnings)

        # 4. diarization ---------------------------------------------------
        started = time.perf_counter()
        _reset_peak()
        if self.skip_diarization or not self.cfg.diarization.enabled:
            diarization = self._stage_diarization(
                prepared, work_dir, state, fingerprints["diarization"]
            )
            if self.progress is not None:
                self.progress.skip_stage("diarization", "Диаризация")
        else:
            with self._stage_progress("diarization", "Диаризация"):
                diarization = self._stage_diarization(
                    prepared, work_dir, state, fingerprints["diarization"]
                )
        timings["diarization"] = round(time.perf_counter() - started, 3)
        peak_diarization = _peak_mb()
        warnings.extend(diarization.warnings)

        # 5. alignment + transcript ---------------------------------------
        started = time.perf_counter()
        with self._stage_progress("alignment", "Выравнивание реплик"):
            transcript = self._stage_alignment(
                prepared,
                asr_result,
                diarization,
                work_dir,
                state,
                meeting_id,
                warnings,
                fingerprints["alignment"],
            )
        timings["alignment"] = round(time.perf_counter() - started, 3)

        # 6. terminology ---------------------------------------------------
        started = time.perf_counter()
        with self._stage_progress("terminology", "Терминология"):
            warnings.extend(
                self._stage_terminology(
                    transcript, work_dir, state, fingerprints["terminology"]
                )
            )
        timings["terminology"] = round(time.perf_counter() - started, 3)

        if self.speaker_map:
            applied = apply_speaker_names(transcript, self.speaker_map)
            warnings.append(f"applied display names for {applied} speaker(s)")

        # 7. optional LLM report ------------------------------------------
        started = time.perf_counter()
        if self.skip_llm or not self.cfg.llm.enabled:
            report, report_warnings = self._stage_report(
                transcript,
                visual.cards,
                work_dir,
                state,
                fingerprints["llm_report"],
            )
            if self.progress is not None:
                self.progress.skip_stage("llm_report", "LLM-отчёт")
        else:
            with self._stage_progress("llm_report", "LLM-отчёт"):
                if self.before_report is not None:
                    self.before_report()
                try:
                    report, report_warnings = self._stage_report(
                        transcript,
                        visual.cards,
                        work_dir,
                        state,
                        fingerprints["llm_report"],
                    )
                finally:
                    if self.after_report is not None:
                        self.after_report()
        timings["llm_report"] = round(time.perf_counter() - started, 3)
        warnings.extend(report_warnings)

        # 8. persist + outputs --------------------------------------------
        started = time.perf_counter()
        with self._stage_progress("outputs", "Сохранение результатов"):
            transcript.config_snapshot = self.cfg.to_dict()
            transcript.run = run
            transcript.warnings = _dedupe(warnings)
            transcript.run.finished_at = _now()
            transcript.run.stage_timings_seconds = timings
            transcript.run.vram_peak_mb = max(
                [value for value in (peak_asr, peak_diarization) if value is not None],
                default=None,
            )
            persistence.save_transcript(transcript, work_dir / "transcript.json")
            state_mod.save_state(work_dir, state)

            outputs = self._write_outputs(transcript, report, work_dir)
        timings["outputs"] = round(time.perf_counter() - started, 3)
        return RunResult(
            transcript=transcript,
            outputs=outputs,
            report=report,
            work_dir=work_dir,
            visual=visual,
        )

    # -- stages --------------------------------------------------------------
    @contextmanager
    def _stage_progress(self, key: str, label: str):
        if self.progress is not None:
            self.progress.start_stage(key, label)
        try:
            yield
        except BaseException:
            if self.progress is not None:
                self.progress.fail_stage(key, label)
            raise
        else:
            if self.progress is not None:
                self.progress.finish_stage(key, label)

    def _stage_fingerprints(self, source_identity: str) -> dict[str, str]:
        cfg = self.cfg
        plain = cfg.to_dict()

        def fp(payload: dict[str, Any]) -> str:
            return state_mod.config_fingerprint(payload)

        glossary_sources: list[list[Any]] = []
        terminology_dir = cfg.resolve(cfg.paths.terminology_dir)
        for name in cfg.terminology.profiles:
            path = terminology_dir / f"{name}.yaml"
            glossary_sources.append(
                [str(path), file_sha256(path) if path.is_file() else None]
            )
        for item in cfg.terminology.glossary_files:
            path = cfg.resolve(item)
            glossary_sources.append(
                [str(path), file_sha256(path) if path.is_file() else None]
            )

        preprocess = fp(
            {
                "type": "preprocess",
                "source": source_identity,
                "audio": plain["audio"],
                "chunk_seconds": cfg.asr.chunk_seconds,
                "overlap_seconds": cfg.asr.overlap_seconds,
            }
        )
        asr = fp(
            {
                "type": "asr",
                "preprocess": preprocess,
                "asr": plain["asr"],
                "language": cfg.language,
            }
        )
        diarization = fp(
            {
                "type": "diarization",
                "source": source_identity,
                "diarization": plain["diarization"],
                "skip": self.skip_diarization,
            }
        )
        alignment = fp(
            {"type": "alignment", "asr": asr, "diarization": diarization}
        )
        visual = fp(
            {
                "type": "visual",
                "source": source_identity,
                "audio": plain["audio"],
                "visual": plain["visual"],
                "skip": not cfg.visual.enabled,
            }
        )
        terminology = fp(
            {
                "type": "terminology",
                "alignment": alignment,
                "terminology": plain["terminology"],
                "glossaries": glossary_sources,
            }
        )
        llm_report = fp(
            {
                "type": "llm_report",
                "terminology": terminology,
                "visual": visual,
                "llm": plain["llm"],
                "skip": self.skip_llm,
                "speaker_map": (
                    file_sha256(self.speaker_map_path)
                    if self.speaker_map_path and self.speaker_map_path.is_file()
                    else None
                ),
            }
        )
        return {
            "preprocess": preprocess,
            "asr": asr,
            "diarization": diarization,
            "alignment": alignment,
            "visual": visual,
            "terminology": terminology,
            "llm_report": llm_report,
        }

    def _stage_preprocess(
        self, source: Path, work_dir: Path, state: dict[str, Any], fingerprint: str
    ) -> PreparedAudio:
        artifact = work_dir / "preprocess.json"
        if not self.force and state_mod.stage_done(
            state, "preprocess", artifact, fingerprint
        ):
            data = json.loads(artifact.read_text(encoding="utf-8"))
            wav_path = Path(data["wav_path"])
            if wav_path.is_file():
                metadata = persistence.audio_metadata_from_dict(data["metadata"])
                windows = plan_chunks(
                    metadata.duration or 0.0,
                    self.cfg.asr.chunk_seconds,
                    self.cfg.asr.overlap_seconds,
                )
                chunks = _build_chunks(wav_path, work_dir / "chunks", windows)
                return PreparedAudio(
                    source=source, wav_path=wav_path, metadata=metadata, chunks=chunks
                )

        prepared = preprocess_audio(
            source,
            work_dir,
            self.cfg.audio,
            chunk_seconds=self.cfg.asr.chunk_seconds,
            overlap_seconds=self.cfg.asr.overlap_seconds,
        )
        payload = {
            "wav_path": str(prepared.wav_path),
            "metadata": prepared.metadata.to_dict(),
            "chunks": [
                {"index": c.index, "start": c.start, "end": c.end, "path": str(c.path)}
                for c in prepared.chunks
            ],
        }
        artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        state_mod.mark_done(state, "preprocess", artifact, fingerprint)
        return prepared

    def _stage_visual(
        self,
        source: Path,
        prepared: PreparedAudio,
        work_dir: Path,
        state: dict[str, Any],
        fingerprint: str,
    ) -> VisualResult:
        if not self.cfg.visual.enabled or not prepared.metadata.has_video:
            warnings: list[str] = []
            if self.cfg.visual.enabled and not prepared.metadata.has_video:
                warnings.append(
                    "visual analysis was requested but the input has no video stream"
                )
            return VisualResult(cards=[], stats={"enabled": False}, warnings=warnings, enabled=False)
        artifact = work_dir / "visual.json"
        if not self.force and state_mod.stage_done(
            state, "visual", artifact, fingerprint
        ):
            cached = load_visual_result(work_dir)
            if cached is not None:
                return cached
        result = extract_visual(
            source,
            work_dir,
            self.cfg.visual,
            ffmpeg=self.cfg.audio.ffmpeg,
            duration=prepared.metadata.duration or 0.0,
        )
        artifact.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if result.enabled:
            state_mod.mark_done(state, "visual", artifact, fingerprint)
        return result

    def _stage_asr(
        self, prepared: PreparedAudio, work_dir: Path, state: dict[str, Any], fingerprint: str
    ) -> ASRResult:
        artifact = work_dir / "asr.json"
        if not self.force and state_mod.stage_done(state, "asr", artifact, fingerprint):
            return persistence.load_asr_result(artifact)

        if self.cfg.asr.backend != "whisper":
            raise ConfigError(
                f"unsupported asr.backend '{self.cfg.asr.backend}'; supported: whisper"
            )
        transcriber: Transcriber = WhisperBackend(
            model_dir=self.cfg.models_path(self.cfg.asr.model_dir),
            device=self.cfg.asr.device,
            compute_type=self.cfg.asr.compute_type,
            cpu_threads=self.cfg.asr.cpu_threads,
            beam_size=self.cfg.asr.beam_size,
            word_timestamps=self.cfg.asr.word_timestamps,
            vad_filter=self.cfg.asr.vad_filter,
            vad_min_silence_ms=self.cfg.asr.vad_min_silence_ms,
            condition_on_previous_text=self.cfg.asr.condition_on_previous_text,
            local_files_only=self.cfg.asr.local_files_only,
        )
        try:
            result = transcriber.transcribe(
                prepared.chunks,
                language=self.cfg.language,
                initial_prompt=self.cfg.asr.initial_prompt or None,
            )
        finally:
            transcriber.release()
            _release_torch()
        persistence.save_asr_result(result, artifact)
        state_mod.mark_done(state, "asr", artifact, fingerprint)
        return result

    def _stage_diarization(
        self, prepared: PreparedAudio, work_dir: Path, state: dict[str, Any], fingerprint: str
    ) -> DiarizationResult:
        artifact = work_dir / "diarization.json"
        if self.skip_diarization or not self.cfg.diarization.enabled:
            return DiarizationResult(
                warnings=["diarization disabled; segments have no speaker_id"]
            )
        if not self.force and state_mod.stage_done(
            state, "diarization", artifact, fingerprint
        ):
            return persistence.load_diarization_result(artifact)

        if self.cfg.diarization.backend != "nemo":
            raise ConfigError(
                f"unsupported diarization.backend '{self.cfg.diarization.backend}'; supported: nemo"
            )
        diarizer: Diarizer = NemoDiarizer(
            speaker_model_path=self.cfg.models_path(
                "speakerverification_en_titanet_large",
                self.cfg.diarization.speaker_model,
            ),
            work_dir=work_dir,
            cfg=self.cfg.diarization,
        )
        try:
            result = diarizer.diarize(
                prepared.wav_path, num_speakers=self.cfg.diarization.num_speakers
            )
        except TranscriberError as exc:
            if not self.cfg.pipeline.continue_on_diarization_error:
                raise
            # Do not mark a failed attempt as completed: the next run must retry
            # instead of silently reusing an empty result.
            return DiarizationResult(
                failed=True,
                warnings=[f"diarization failed and was skipped: {exc}"],
            )
        finally:
            diarizer.release()
            _release_torch()

        persistence.save_diarization_result(result, artifact)
        state_mod.mark_done(state, "diarization", artifact, fingerprint)
        return result

    def _stage_alignment(
        self,
        prepared: PreparedAudio,
        asr_result: ASRResult,
        diarization: DiarizationResult,
        work_dir: Path,
        state: dict[str, Any],
        meeting_id: str,
        warnings: list[str],
        fingerprint: str,
    ) -> Transcript:
        artifact = work_dir / "transcript_aligned.json"
        if not self.force and state_mod.stage_done(
            state, "alignment", artifact, fingerprint
        ):
            return persistence.load_transcript(artifact)

        segments, speakers, align_warnings = align_segments(
            [persistence.segment_from_dict(s.to_dict()) for s in asr_result.segments],
            diarization.intervals,
        )
        warnings.extend(align_warnings)
        transcript = Transcript(
            meeting_id=meeting_id,
            language=asr_result.language or self.cfg.language,
            source_audio=prepared.metadata,
            segments=segments,
            speakers=speakers,
            models=self._model_info(asr_result, diarization),
            config_snapshot=self.cfg.to_dict(),
            diarization_intervals=diarization.intervals,
        )
        if diarization.failed:
            return transcript
        persistence.save_transcript(transcript, artifact)
        state_mod.mark_done(state, "alignment", artifact, fingerprint)
        return transcript

    def _stage_terminology(
        self, transcript: Transcript, work_dir: Path, state: dict[str, Any], fingerprint: str
    ) -> list[str]:
        if not self.cfg.terminology.enabled:
            for segment in transcript.segments:
                segment.corrected_text = segment.raw_text
            return []
        artifact = work_dir / "transcript_corrected.json"
        if not self.force and state_mod.stage_done(
            state, "terminology", artifact, fingerprint
        ):
            corrected = persistence.load_transcript(artifact)
            transcript.segments = corrected.segments
            transcript.speakers = corrected.speakers
            return []
        try:
            corrector = TerminologyCorrector.from_config(
                self.cfg.terminology,
                self.cfg.resolve(self.cfg.paths.terminology_dir),
                [self.cfg.resolve(p) for p in self.cfg.terminology.glossary_files],
            )
        except TranscriberError as exc:
            for segment in transcript.segments:
                segment.corrected_text = segment.raw_text
            return [f"terminology correction skipped: {exc}"]
        warnings = corrector.correct_segments(transcript.segments)
        persistence.save_transcript(transcript, artifact)
        state_mod.mark_done(state, "terminology", artifact, fingerprint)
        return warnings

    def _stage_report(
        self,
        transcript: Transcript,
        cards: list[CodeCard],
        work_dir: Path,
        state: dict[str, Any],
        fingerprint: str,
    ) -> tuple[dict[str, Any] | None, list[str]]:
        if self.skip_llm or not self.cfg.llm.enabled:
            return None, []
        artifact = work_dir / "report.json"
        if not self.force and state_mod.stage_done(
            state, "llm_report", artifact, fingerprint
        ):
            return json.loads(artifact.read_text(encoding="utf-8")), []
        report, warnings = generate_meeting_report(
            transcript,
            self.cfg.llm,
            work_dir=work_dir,
            code_cards=cards,
            visual_context_chars=self.cfg.visual.report_context_chars,
            visual_align_window=self.cfg.visual.align_window_seconds,
        )
        if report is not None:
            artifact.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            state_mod.mark_done(state, "llm_report", artifact, fingerprint)
        return report, warnings

    # -- outputs -------------------------------------------------------------
    def _write_outputs(
        self,
        transcript: Transcript,
        report: dict[str, Any] | None,
        work_dir: Path,
    ) -> dict[str, Path]:
        output_dir = self.cfg.resolve(self.cfg.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs: dict[str, Path] = {}
        overwrite = self.force or self.cfg.output.overwrite
        frames_rel = f"{transcript.meeting_id}.frames"
        if report is not None:
            _copy_report_frames(report, work_dir / "frames", output_dir / frames_rel)
        for fmt in self.cfg.output.formats:
            path = _unique_path(output_dir / f"{transcript.meeting_id}.{fmt}", overwrite)
            if fmt == "json":
                outputs["json"] = write_json(
                    transcript, path, include_words=self.cfg.output.write_words
                )
            elif fmt == "markdown":
                outputs["markdown"] = write_markdown(transcript, path, report)
            elif fmt == "srt":
                outputs["srt"] = write_srt(transcript, path)
        visual_path = work_dir / "visual.json"
        if self.cfg.visual.enabled and visual_path.is_file():
            target = _unique_path(
                output_dir / f"{transcript.meeting_id}.visual.json", overwrite
            )
            shutil.copy2(visual_path, target)
            outputs["visual_json"] = target
        if report is not None:
            report_json = _unique_path(
                output_dir / f"{transcript.meeting_id}.report.json", overwrite
            )
            report_json.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            report_md = _unique_path(
                output_dir / f"{transcript.meeting_id}.report.md", overwrite
            )
            report_md.write_text(
                render_report_markdown(report, frames_dir=frames_rel), encoding="utf-8"
            )
            outputs["report_json"] = report_json
            outputs["report_markdown"] = report_md
        return outputs

    def _model_info(
        self, asr: ASRResult, diarization: DiarizationResult
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "asr": asr.model,
            "asr_backend": asr.backend,
            "diarization": diarization.model or "disabled",
            "diarization_backend": diarization.backend or "disabled",
        }
        manifest = _manifest_info(self.cfg)
        if manifest:
            info["manifest"] = manifest
        return info


# -- helpers -----------------------------------------------------------------
def _copy_report_frames(
    report: dict[str, Any], source_dir: Path, target_dir: Path
) -> None:
    """Copy only frames referenced by the report next to the Markdown file."""

    frames = {
        str(card.get("frame"))
        for card in report.get("code_cards") or []
        if isinstance(card, dict) and card.get("frame")
    }
    if not frames:
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in frames:
        source = source_dir / name
        if source.is_file():
            shutil.copy2(source, target_dir / name)


def _build_chunks(
    wav_path: Path, chunk_dir: Path, windows: list[tuple[float, float]]
) -> list[Any]:
    from ..audio.preprocess import AudioChunk

    if len(windows) == 1 and windows[0][0] == 0.0:
        return [AudioChunk(index=0, start=0.0, end=windows[0][1], path=wav_path)]
    chunks = []
    for index, (start, end) in enumerate(windows):
        path = chunk_dir / f"chunk_{index:04d}.wav"
        if not path.is_file():
            raise PipelineError(
                f"cached audio chunk is missing: {path}. Re-run with --force."
            )
        chunks.append(AudioChunk(index=index, start=start, end=end, path=path))
    return chunks


def _unique_path(path: Path, force: bool) -> Path:
    if force or not path.exists():
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{stamp}_{counter}{path.suffix}")
        counter += 1
    return candidate


def _manifest_info(cfg: AppConfig) -> dict[str, Any]:
    from ..models import iter_groups, load_manifest

    candidates = [
        cfg.resolve("models/manifest.yaml"),
        cfg.models_path("manifest.yaml"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            try:
                manifest = load_manifest(candidate)
            except TranscriberError:
                return {}
            return {
                group.key: {
                    "name": group.name,
                    "repo": group.repo,
                    "revision": group.revision,
                    "license": group.license,
                }
                for group in iter_groups(manifest)
            }
    return {}


def _library_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    result: dict[str, str] = {}
    for package in (
        "faster-whisper",
        "ctranslate2",
        "torch",
        "nemo_toolkit",
        "silero-vad",
        "numpy",
        "PyYAML",
    ):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            continue
    return result


def _reset_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        return


def _peak_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024**2), 1)
    except ImportError:
        return None
    return None


def _release_torch() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
