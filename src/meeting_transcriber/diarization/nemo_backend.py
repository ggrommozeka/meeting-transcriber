"""NeMo ``ClusteringDiarizer`` backend.

Cascade: Silero VAD -> NeMo TitaNet-Large speaker embeddings -> spectral
clustering. NeMo does not hard-limit the number of speakers; ``max_speakers``
is an upper bound for automatic speaker counting.

Offline guarantees:
* the TitaNet checkpoint is loaded from a local ``.nemo`` file;
* no NGC/Hugging Face resolution is attempted;
* NeMo writes intermediate files only inside the meeting work directory.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..config import DiarizationConfig
from ..errors import BackendUnavailableError, ModelNotFoundError
from ..types import DiarizationInterval
from .base import DiarizationResult, Diarizer
from .silero_vad import run_silero_vad

RTTM_SPEAKER = "SPEAKER"


class NemoDiarizer(Diarizer):
    name = "nemo-clustering"

    def __init__(
        self,
        speaker_model_path: str | Path,
        work_dir: str | Path,
        cfg: DiarizationConfig,
    ) -> None:
        self.speaker_model_path = Path(speaker_model_path).expanduser()
        self.work_dir = Path(work_dir).expanduser()
        self.cfg = cfg
        self._loaded = False

    # -- lifecycle -----------------------------------------------------------
    def load(self) -> None:
        if self._loaded:
            return
        if not self.speaker_model_path.is_file():
            raise ModelNotFoundError(
                f"speaker embedding checkpoint not found: {self.speaker_model_path}. "
                "Run ./scripts/download_models.sh (runtime never downloads weights)."
            )
        try:
            import nemo.collections.asr  # noqa: F401
            from nemo.collections.asr.models import ClusteringDiarizer  # noqa: F401
        except ImportError as exc:
            raise BackendUnavailableError(
                "NeMo ASR toolkit is not installed. Install the diarization extras: "
                "pip install -r requirements/diarization.txt"
            ) from exc
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self._loaded = True

    def release(self) -> None:
        self._loaded = False
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # -- inference -----------------------------------------------------------
    def diarize(
        self,
        wav_path: str | Path,
        num_speakers: int | None = None,
    ) -> DiarizationResult:
        if not self._loaded:
            self.load()

        from nemo.collections.asr.models import ClusteringDiarizer
        from omegaconf import OmegaConf

        wav = Path(wav_path).expanduser().resolve()
        meeting_id = wav.stem
        run_dir = self.work_dir / "diarization"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        result = DiarizationResult(backend=self.name, model=str(self.speaker_model_path))
        speech = run_silero_vad(wav, self.cfg)
        if not speech:
            result.warnings.append("VAD found no speech; diarization was skipped")
            return result

        recording_duration = max(end for _, end in speech)
        scales_cfg = self.cfg
        if recording_duration >= self.cfg.long_audio_threshold_seconds > 0:
            scales_cfg = replace(
                self.cfg,
                window_lengths=self.cfg.long_audio_window,
                shift_lengths=self.cfg.long_audio_shift,
                multiscale_weights=[1],
            )
            result.warnings.append(
                "long recording: single-scale diarization "
                f"({self.cfg.long_audio_window}s/{self.cfg.long_audio_shift}s) "
                "to bound clustering memory"
            )
            result.info["embedding_scales"] = "single"

        vad_manifest = run_dir / "vad_manifest.json"
        write_vad_manifest(vad_manifest, wav, meeting_id, speech)

        main_manifest = run_dir / "manifest.json"
        write_main_manifest(main_manifest, wav, meeting_id, num_speakers)

        cfg = build_diarizer_config(
            cfg=scales_cfg,
            manifest_path=main_manifest,
            vad_manifest_path=vad_manifest,
            speaker_model_path=self.speaker_model_path,
            out_dir=run_dir / "nemo_out",
        )
        try:
            diarizer = ClusteringDiarizer(cfg=OmegaConf.create(cfg))
            diarizer.diarize()
        except Exception as exc:
            raise BackendUnavailableError(f"NeMo diarization failed: {exc}") from exc
        finally:
            self.release()

        rttm_path = run_dir / "nemo_out" / "pred_rttms" / f"{meeting_id}.rttm"
        if not rttm_path.is_file():
            raise BackendUnavailableError(
                f"NeMo finished without producing the expected RTTM file: {rttm_path}"
            )

        intervals = parse_rttm(rttm_path)
        result.intervals = normalize_speaker_ids(intervals)
        result.info["num_speakers"] = len(result.speaker_ids)
        result.info["vad_segments"] = len(speech)
        if not self.cfg.keep_intermediate:
            shutil.rmtree(run_dir / "nemo_out" / "speaker_outputs", ignore_errors=True)
            shutil.rmtree(run_dir / "nemo_out" / "vad_outputs", ignore_errors=True)
        return result


# -- pure helpers (unit-tested without NeMo) --------------------------------
def write_vad_manifest(
    path: Path,
    wav_path: Path,
    meeting_id: str,
    speech: list[tuple[float, float]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for start, end in speech:
            handle.write(
                json.dumps(
                    {
                        "uniq_id": meeting_id,
                        "audio_filepath": str(wav_path),
                        "offset": round(start, 3),
                        "duration": round(end - start, 3),
                        "label": "infer",
                        "text": "-",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_main_manifest(
    path: Path,
    wav_path: Path,
    meeting_id: str,
    num_speakers: int | None,
) -> None:
    entry = {
        "uniq_id": meeting_id,
        "audio_filepath": str(wav_path),
        "offset": 0.0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "num_speakers": num_speakers,
        "rttm_filepath": None,
    }
    path.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")


def build_diarizer_config(
    cfg: DiarizationConfig,
    manifest_path: Path,
    vad_manifest_path: Path,
    speaker_model_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    oracle = cfg.num_speakers is not None
    windows = cfg.window_lengths
    shifts = cfg.shift_lengths
    weights = cfg.multiscale_weights
    single_scale = isinstance(windows, (int, float)) or isinstance(shifts, (int, float))
    if single_scale:
        embedding_params = {
            "window_length_in_sec": float(windows),
            "shift_length_in_sec": float(shifts),
            "multiscale_weights": None,
            "save_embeddings": False,
        }
    else:
        embedding_params = {
            "window_length_in_sec": list(windows),
            "shift_length_in_sec": list(shifts),
            "multiscale_weights": list(weights),
            "save_embeddings": False,
        }

    return {
        "name": "cluster_diarizer",
        "device": cfg.device if cfg.device != "auto" else "cuda",
        "sample_rate": 16000,
        "num_workers": 0,
        "batch_size": 64,
        "verbose": False,
        "diarizer": {
            "manifest_filepath": str(manifest_path),
            "out_dir": str(out_dir),
            "oracle_vad": False,
            "collar": 0.25,
            "ignore_overlap": True,
            "vad": {
                "model_path": None,
                "external_vad_manifest": str(vad_manifest_path),
                "parameters": {
                    "window_length_in_sec": 0.15,
                    "shift_length_in_sec": 0.01,
                    "smoothing": "median",
                    "overlap": 0.875,
                    "onset": 0.4,
                    "offset": 0.7,
                    "pad_onset": 0.05,
                    "pad_offset": -0.1,
                    "min_duration_on": 0.2,
                    "min_duration_off": 0.2,
                    "filter_speech_first": True,
                },
            },
            "speaker_embeddings": {
                "model_path": str(speaker_model_path),
                "parameters": embedding_params,
            },
            "clustering": {
                "parameters": {
                    "oracle_num_speakers": oracle,
                    "max_num_speakers": cfg.max_speakers,
                    "enhanced_count_thres": cfg.enhanced_count_thres,
                    "max_rp_threshold": cfg.max_rp_threshold,
                    "sparse_search_volume": cfg.sparse_search_volume,
                }
            },
        },
    }


def parse_rttm(path: Path) -> list[DiarizationInterval]:
    intervals: list[DiarizationInterval] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if not parts or parts[0] != RTTM_SPEAKER:
            continue
        if len(parts) < 8:
            raise BackendUnavailableError(f"{path}:{lineno}: malformed RTTM line")
        start = float(parts[3])
        duration = float(parts[4])
        speaker = parts[7]
        if duration <= 0:
            continue
        intervals.append(
            DiarizationInterval(
                start=round(start, 3),
                end=round(start + duration, 3),
                speaker_id=speaker,
            )
        )
    intervals.sort(key=lambda i: (i.start, i.end, i.speaker_id))
    return intervals


def normalize_speaker_ids(intervals: list[DiarizationInterval]) -> list[DiarizationInterval]:
    """Map backend labels to stable ``speaker_00`` ids by first appearance."""

    mapping: dict[str, str] = {}
    for interval in intervals:
        if interval.speaker_id not in mapping:
            mapping[interval.speaker_id] = f"speaker_{len(mapping):02d}"
    return [
        DiarizationInterval(
            start=i.start,
            end=i.end,
            speaker_id=mapping[i.speaker_id],
            confidence=i.confidence,
        )
        for i in intervals
    ]
