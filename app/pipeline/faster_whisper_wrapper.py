"""FasterWhisperWrapper: ModelWrapper backed by faster-whisper, for CPU and GPU alike.

Design decisions
────────────────
* Device and precision are run settings, not code paths.
  `model_options.device` ("cpu" | "cuda") and `model_options.compute_type` ("int8", "float16", ...) are validated by
  FasterWhisperOptions and stored in the run's resolved config, so they are part of config_hash. The defaults come
  from the submitting machine's settings (DEVICE, COMPUTE_TYPE). Before Part H a separate gpu/ package copied this
  module to change those two values, and CPU and GPU runs got the same hash.

* Decoding follows the stored config exactly.
  beam_size, language, task AND temperature are passed to faster-whisper. A single temperature disables
  faster-whisper's fallback schedule (0.0, 0.2, ... 1.0); before Part H temperature was never passed, so the
  fallback was active although the stored config said 0.0.

* needs_preparation = True
  The worker sends the cached 16 kHz mono WAV stream and a Prepared object (chunk time boundaries). We read each
  slice, transcribe it, and stitch the results.

* Process-level model cache
  Loading takes 1 to 5 s. One model per (size, device, compute_type) per process, lock-guarded. In Celery each child
  process loads it on its first task.

* Chunk slicing and stitching
  We seek to each chunk's sample range and pass a WAV slice to faster-whisper, so one chunk is in memory at a time.
  Chunk transcripts are joined with a single space. VAD chunks do not overlap, so no de-duplication is needed.
"""
from __future__ import annotations

import io
import logging
import threading
from typing import BinaryIO, Literal

from pydantic import BaseModel, create_model, model_validator

from app.pipeline.model_wrapper import ModelWrapper
from app.pipeline.prepare import Prepared
from app.pipeline.run_config import RunConfig, Strict

log = logging.getLogger("faster_whisper")

Device = Literal["cpu", "cuda"]
ComputeType = Literal["int8", "int8_float32", "int8_float16", "int8_bfloat16", "float16", "bfloat16", "float32"]
_GPU_ONLY = {"float16", "int8_float16"}  # CTranslate2 has no float16 kernels on CPU


class FasterWhisperOptions(Strict):
    """config.model_options for every faster-whisper model."""

    device: Device = "cpu"
    compute_type: ComputeType = "int8"

    @model_validator(mode="after")
    def _valid_combination(self):
        if self.device == "cpu" and self.compute_type in _GPU_ONLY:
            raise ValueError(f"compute_type '{self.compute_type}' needs device 'cuda'")
        return self


def faster_whisper_options(device: str, compute_type: str) -> type[BaseModel]:
    """FasterWhisperOptions whose defaults are this machine's settings.

    The defaults are filled into every stored config, so the hash records the device and precision a run used
    even when the client did not name them."""
    FasterWhisperOptions(device=device, compute_type=compute_type)  # fail fast on a bad DEVICE/COMPUTE_TYPE pair
    return create_model(
        "FasterWhisperOptions",
        __base__=FasterWhisperOptions,
        device=(Device, device),
        compute_type=(ComputeType, compute_type),
    )


# ── process-level model cache ──────────────────────────────────────────────────
_lock = threading.Lock()
_models: dict[str, object] = {}  # "<model_size>/<device>/<compute_type>" -> WhisperModel


def _get_model(model_size: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel  # heavy import; only workers that run the model pay for it

    cache_key = f"{model_size}/{device}/{compute_type}"
    if cache_key not in _models:
        with _lock:
            if cache_key not in _models:
                log.info("loading faster-whisper model",
                         extra={"model_size": model_size, "device": device, "compute_type": compute_type})
                _models[cache_key] = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _models[cache_key]


def _read_chunk_bytes(wav_stream: BinaryIO, start_s: float, end_s: float, sample_rate: int) -> bytes:
    """One time slice of an open WAV stream, returned as WAV bytes at the same sample rate."""
    import soundfile as sf

    start_frame = int(start_s * sample_rate)
    n_frames = max(1, int((end_s - start_s) * sample_rate))
    wav_stream.seek(0)
    with sf.SoundFile(wav_stream) as f:
        f.seek(start_frame)
        data = f.read(n_frames, dtype="float32", always_2d=False)
    buf = io.BytesIO()
    with sf.SoundFile(buf, mode="w", samplerate=sample_rate, channels=1, subtype="PCM_16", format="WAV") as out:
        out.write(data)
    return buf.getvalue()


class FasterWhisperWrapper(ModelWrapper):
    """faster-whisper transcriber, VAD-chunked, with chunk-and-stitch.

    device and compute_type given here are only the fallback for legacy runs whose stored config has no
    model_options; new runs always carry both."""

    needs_preparation: bool = True

    def __init__(self, model_size: str = "tiny", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

    def resolve_hardware(self, config: RunConfig) -> tuple[str, str]:
        opts = config.model_options or {}
        return opts.get("device", self.device), opts.get("compute_type", self.compute_type)

    def transcribe(self, audio: BinaryIO, config: RunConfig, prepared: Prepared | None = None) -> str:
        device, compute_type = self.resolve_hardware(config)
        model = _get_model(self.model_size, device, compute_type)
        dec = config.decoding
        kwargs = {"language": dec.language, "beam_size": dec.beam_size, "task": dec.task,
                  "temperature": dec.temperature}
        sample_rate = prepared.sample_rate if prepared else config.audio.sample_rate
        chunks = prepared.chunks if prepared else None

        if not chunks:  # no chunk plan (or an empty one): transcribe the whole file in one shot
            audio.seek(0)
            return self._transcribe_stream(model, audio, **kwargs)

        parts: list[str] = []
        for chunk in chunks:
            chunk_buf = io.BytesIO(_read_chunk_bytes(audio, chunk.start, chunk.end, sample_rate))
            text = self._transcribe_stream(model, chunk_buf, **kwargs)
            if text:
                parts.append(text)
        return " ".join(parts)

    @staticmethod
    def _transcribe_stream(model, stream: BinaryIO, *, language: str | None, beam_size: int, task: str,
                           temperature: float) -> str:
        stream.seek(0)
        segments, _info = model.transcribe(
            stream,
            language=language,
            beam_size=beam_size,
            task=task,
            temperature=temperature,  # one value = no fallback schedule; matches the stored config
            vad_filter=False,         # our own VAD chunking runs upstream
            word_timestamps=False,
        )
        return " ".join(seg.text.strip() for seg in segments if seg.text.strip())
