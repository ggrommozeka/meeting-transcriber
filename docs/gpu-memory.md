# GPU memory and sequential execution

## Target

Single NVIDIA GPU with 8 GB VRAM. The pipeline never holds ASR and diarization
models at the same time.

## Sequence

1. Load faster-whisper (CTranslate2, `float16`).
2. Transcribe all chunks.
3. Drop the model, `gc.collect()`, `torch.cuda.empty_cache()`.
4. Load TitaNet-Large inside `ClusteringDiarizer`.
5. Diarize, then drop the diarizer and empty the cache.
6. Align on CPU.

`WhisperBackend.release()` and `NemoDiarizer.release()` implement steps 3/5.
The CLI reports `peak VRAM` measured with
`torch.cuda.max_memory_allocated()` per stage.

## Practical defaults

| Setting | Default | Notes |
| --- | --- | --- |
| `asr.compute_type` | `float16` | use `int8_float16` on very tight GPUs |
| `asr.beam_size` | `5` | `1` is faster, slightly less accurate |
| `asr.chunk_seconds` | `900` | 15-minute chunks keep long recordings resumable |
| `asr.overlap_seconds` | `15` | prevents lost words at chunk boundaries |
| `diarization.max_speakers` | `20` | upper bound, not a forced count |
| `diarization.window_lengths` | `[1.5 … 0.5]` | multiscale embeddings for recordings up to 30 min |
| `diarization.long_audio_threshold_seconds` | `1800` | above this, single-scale `1.5s/0.75s` embeddings are used to bound the O(N²) clustering matrix |

## Measured on the reference host

RTX 4060 8 GB, driver 590.48.01, torch 2.6.0+cu124, faster-whisper 1.1.1,
NeMo 2.6.2, 87-minute WebM (Opus, stereo):

| Stage | Result |
| --- | --- |
| ASR + diarization + terminology | 2m31s wall time |
| Peak VRAM (`torch.cuda.max_memory_allocated`) | **4612 MiB** |
| Speakers found | **8** |
| Segments | 945 |
| Diarization mode | long-recording single-scale 1.5s/0.75s |

ASR and diarization models were never resident at the same time; the peak is the
larger of the two stages.

The optional LLM report (unsloth Qwen3.5-9B UD-Q4_K_XL via llama.cpp) took
5m48s for the 20-chunk map pass; consolidating topics and generating the summary
from cached chunks took 2m21s. `llama-server` holds about 6.7 GiB with an 8k
context, so it must not be loaded while whisper is running. The pipeline now
starts it lazily for the report stage, after ASR and diarization have released
their models, and stops it when the report finishes. A server left over from a
previous run is stopped automatically before ASR; `scripts/stop_llama_server.sh`
remains available for manual cleanup.

## If you hit OOM

1. `--device cpu` for diarization experiments.
2. `asr.compute_type: int8_float16`.
3. Reduce `asr.beam_size` to 1.
4. Reduce `asr.chunk_seconds` (does not change per-chunk model memory, but
   lowers peak host/GPU transfer pressure).
5. Reduce `diarization.window_lengths` to `[1.5, 1.0, 0.5]` and matching shifts.
6. Do not run two pipeline instances on the same GPU.

`torch.cuda.empty_cache()` alone is not treated as a memory strategy: model
references are dropped first, then the cache is emptied.

## Measuring the real peak

```bash
meeting-transcriber transcribe data/input/sample.wav --force
# peak VRAM:  XXXX MiB
```

For a more detailed trace:

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1
```

Record the measured value in `docs/known-limitations.md` before claiming a
specific VRAM figure. This repository intentionally does not publish invented
numbers.

## Larger GPUs

On 16–24 GB GPUs ASR and diarization could stay resident, but the MVP keeps the
sequential design: it is the safer default and matches the 8 GB requirement.
