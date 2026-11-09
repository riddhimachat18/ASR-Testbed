"""FasterWhisperWrapper — GPU edition.

Identical to app/pipeline/faster_whisper_wrapper.py with one key difference:
  the device and compute_type passed to WhisperModel come from GPUSettings
  (device="cuda", compute_type="float16" by default) instead of being hardcoded
  to "cpu" / "int8".

Everything else — the process-level singleton, chunk slicing, stitching logic —
is unchanged so results are directly comparable with the CPU baseline.

Design decisions (same as CPU version)
────────────────────────────────────────
* needs_preparation = True
  The worker sends the cached 16 kHz mono WAV stream and a Prepared object
  (list of Chunk time boundaries). We read each slice and transcribe, then stitch.

* Process-level model singleton
  WhisperModel loading takes ~3 s for "tiny" even on GPU. We load once per
  process using a lock-guarded module-level dict. In Celery each child process
  loads the model exactly once on its first task.

* Chunk slicing
  We read the WAV header once, then seek to each chunk's sample range to pass
  a BytesIO slice to faster-whisper. One chunk in RAM at a time.

* Stitching
  Chunk transcripts are joined with a single space after stripping whitespace.
"""
from __future__ import annotations

import io
import logging
import threading
from typing import BinaryIO

import soundfile as sf

from app.pipeline.model_wrapper import ModelWrapper
from app.pipeline.prepare import Prepared
from app.pipeline.run_config import RunConfig

log = logging.getLogger("faster_whisper_gpu")

# ── process-level singleton ────────────────────────────────────────────────────
_lock = threading.Lock()
_models: dict[str, object] = {}  # key: "<model_size>/<device>/<compute_type>" -> WhisperModel


def _get_model(model_size: str, device: str, compute_type: str):
    """Load once per (model_size, device, compute_type) combination per process."""
    from faster_whisper import WhisperModel

    cache_key = f"{model_size}/{device}/{compute_type}"
    if cache_key not in _models:
        with _lock:
            if cache_key not in _models:
                log.info(
                    "loading faster-whisper model on GPU",
                    extra={
                        "model_size": model_size,
                        "device": device,
                        "compute_type": compute_type,
                    },
                )
                _models[cache_key] = WhisperModel(
                    model_size,
                    device=device,
                    compute_type=compute_type,
                )
    return _models[cache_key]


def _read_chunk_bytes(
    wav_stream: BinaryIO, start_s: float, end_s: float, sample_rate: int
) -> bytes:
    """Extract one time slice from an open soundfile-compatible WAV stream.
    Returns a valid WAV bytes object at the same sample rate."""
    start_frame = int(start_s * sample_rate)
    n_frames = max(1, int((end_s - start_s) * sample_rate))

    wav_stream.seek(0)
    with sf.SoundFile(wav_stream) as f:
        f.seek(start_frame)
        data = f.read(n_frames, dtype="float32", always_2d=False)

    buf = io.BytesIO()
    with sf.SoundFile(
        buf, mode="w", samplerate=sample_rate, channels=1, subtype="PCM_16", format="WAV"
    ) as out:
        out.write(data)
    buf.seek(0)
    return buf.read()


class FasterWhisperWrapperGPU(ModelWrapper):
    """faster-whisper transcriber running on GPU (cuda), VAD-chunked, with chunk-and-stitch.

    The only difference from FasterWhisperWrapper (CPU) is that device and
    compute_type default to GPU-appropriate values and are read from GPUSettings
    when instantiated via the GPU registry.
    """

    needs_preparation: bool = True

    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cuda",         # GPU default (vs "cpu" in the original)
        compute_type: str = "float16", # GPU default (vs "int8" in the original)
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

    def transcribe(
        self,
        audio: BinaryIO,
        config: RunConfig,
        prepared: Prepared | None = None,
    ) -> str:
        model = _get_model(self.model_size, self.device, self.compute_type)

        lang = config.decoding.language
        beam_size = config.decoding.beam_size
        task = config.decoding.task
        sample_rate = prepared.sample_rate if prepared else config.audio.sample_rate
        chunks = prepared.chunks if prepared else None

        if not chunks:
            audio.seek(0)
            return self._transcribe_stream(model, audio, lang, beam_size, task)

        parts: list[str] = []
        for chunk in chunks:
            chunk_bytes = _read_chunk_bytes(audio, chunk.start, chunk.end, sample_rate)
            chunk_buf = io.BytesIO(chunk_bytes)
            text = self._transcribe_stream(model, chunk_buf, lang, beam_size, task)
            if text:
                parts.append(text)

        return " ".join(parts)

    @staticmethod
    def _transcribe_stream(
        model,
        stream: BinaryIO,
        language: str | None,
        beam_size: int,
        task: str,
    ) -> str:
        """Transcribe a single WAV stream; return the joined segment text."""
        stream.seek(0)
        segments, _info = model.transcribe(
            stream,
            language=language,
            beam_size=beam_size,
            task=task,
            vad_filter=False,       # our own VAD chunking is done upstream
            word_timestamps=False,
        )
        return " ".join(seg.text.strip() for seg in segments if seg.text.strip())
