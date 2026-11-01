"""FasterWhisperWrapper — ModelWrapper backed by faster-whisper.

Design decisions
────────────────
* needs_preparation = True
  The worker sends us the cached 16 kHz mono WAV stream and a Prepared object
  (list of Chunk time boundaries). We read each slice from the stream and
  transcribe it, then stitch the results.

* Process-level model singleton
  WhisperModel loading takes 2-5 seconds and ~75 MB RAM for 'tiny'. We load it
  once per *process* (not per request, not per task) using a lock-guarded
  module-level variable. In Celery, each child process loads the model exactly
  once on its first task, then reuses it for all subsequent tasks in that
  process lifetime.

* Chunk slicing
  We read the WAV header once, then seek to each chunk's sample range to pass
  a BytesIO slice to faster-whisper. This avoids loading the full file into RAM
  for very large files — only one chunk is in memory at a time.

* Stitching
  Chunk transcripts are joined with a single space after stripping leading/
  trailing whitespace. No overlap handling is needed because VAD chunking
  produces non-overlapping segments (fixed chunking with overlap_s=0 is the
  default for this wrapper).
"""
from __future__ import annotations

import io
import logging
import threading
from typing import BinaryIO

import numpy as np
import soundfile as sf

from app.pipeline.model_wrapper import ModelWrapper
from app.pipeline.prepare import Prepared
from app.pipeline.run_config import RunConfig

log = logging.getLogger("faster_whisper")

# ── process-level singleton ────────────────────────────────────────────────────
_lock = threading.Lock()
_models: dict[str, object] = {}  # key: "<model_size>/<device>" -> WhisperModel


def _get_model(model_size: str, device: str, compute_type: str):
    """Load once per (model_size, device) combination per process."""
    from faster_whisper import WhisperModel

    cache_key = f"{model_size}/{device}/{compute_type}"
    if cache_key not in _models:
        with _lock:
            if cache_key not in _models:
                log.info(
                    "loading faster-whisper model",
                    extra={"model_size": model_size, "device": device, "compute_type": compute_type},
                )
                _models[cache_key] = WhisperModel(
                    model_size,
                    device=device,
                    compute_type=compute_type,
                )
    return _models[cache_key]


def _read_chunk_bytes(wav_stream: BinaryIO, start_s: float, end_s: float, sample_rate: int) -> bytes:
    """Extract one time slice from an open soundfile-compatible WAV stream.
    Returns a valid WAV bytes object at the same sample rate."""
    start_frame = int(start_s * sample_rate)
    n_frames = max(1, int((end_s - start_s) * sample_rate))

    wav_stream.seek(0)
    with sf.SoundFile(wav_stream) as f:
        f.seek(start_frame)
        data = f.read(n_frames, dtype="float32", always_2d=False)

    buf = io.BytesIO()
    with sf.SoundFile(buf, mode="w", samplerate=sample_rate, channels=1,
                      subtype="PCM_16", format="WAV") as out:
        out.write(data)
    buf.seek(0)
    return buf.read()


class FasterWhisperWrapper(ModelWrapper):
    """faster-whisper backed transcriber, VAD-chunked, with chunk-and-stitch."""

    needs_preparation: bool = True

    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
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

        # Decode config
        lang = config.decoding.language  # None = auto-detect
        beam_size = config.decoding.beam_size
        task = config.decoding.task
        sample_rate = prepared.sample_rate if prepared else config.audio.sample_rate
        chunks = prepared.chunks if prepared else None

        if not chunks:
            # No chunk plan (or empty): transcribe the whole file in one shot.
            audio.seek(0)
            return self._transcribe_stream(model, audio, lang, beam_size, task)

        # Chunk-and-stitch
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
        # faster-whisper accepts a file-like object directly
        segments, _info = model.transcribe(
            stream,
            language=language,
            beam_size=beam_size,
            task=task,
            vad_filter=False,   # we do our own VAD chunking upstream
            word_timestamps=False,
        )
        return " ".join(seg.text.strip() for seg in segments if seg.text.strip())
