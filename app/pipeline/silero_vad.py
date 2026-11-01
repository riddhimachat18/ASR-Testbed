"""Silero-VAD backend for the derived-cache VadBackend protocol.

The model is loaded once per process (lazy singleton) because torch model loading
takes ~1 second and happens inside get_model(), which is called the first time
detect_speech_segments() runs. All subsequent calls reuse the loaded model.

silero-vad 5.x ships its own iterator-based API (get_speech_timestamps); we use
that directly rather than the torchaudio pipeline because it gives us clean
(start, end) float pairs in seconds — exactly what plan_vad_chunks() wants.
"""
from __future__ import annotations

import logging
import threading
from typing import BinaryIO

import silero_vad as _svad
import importlib.metadata

from app.pipeline.run_config import VadConfig

log = logging.getLogger("silero_vad")

# ── version string used as part of the derived-cache key ──────────────────────
try:
    _VERSION = importlib.metadata.version("silero-vad")
except importlib.metadata.PackageNotFoundError:
    _VERSION = "unknown"

# ── process-level singleton ────────────────────────────────────────────────────
_lock: threading.Lock = threading.Lock()
_model = None  # type: ignore[assignment]


def _get_model():
    """Load the Silero-VAD model once per process.

    silero-vad ≤ 5.x: load_silero_vad() returned (model, utils) tuple.
    silero-vad ≥ 6.x: load_silero_vad() returns the model directly.
    We handle both forms so a version downgrade doesn't break things.
    """
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                log.info("loading silero-vad model (first call in this process)")
                result = _svad.load_silero_vad()
                # 6.x returns the model directly; 5.x returned (model, utils)
                _model = result[0] if isinstance(result, tuple) else result
    return _model


class SileroVadBackend:
    """Implements app.pipeline.derived.VadBackend using Silero-VAD."""

    name: str = "silero"
    version: str = _VERSION

    def detect_speech_segments(
        self, audio: BinaryIO, cfg: VadConfig
    ) -> list[tuple[float, float]]:
        """Return a list of (start_s, end_s) speech segments.

        audio must be a seekable stream of a 16 kHz mono WAV (the resampled
        derived artifact from DerivedCache.resampled()). The stream is NOT
        closed here; the caller manages it.
        """
        import soundfile as sf
        import torch

        model = _get_model()

        # Read the entire resampled wav into a float32 tensor.
        # DerivedCache.chunks() guarantees this is already 16 kHz mono PCM_16.
        data, sr = sf.read(audio, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        wav_tensor = torch.from_numpy(data)

        # get_speech_timestamps returns a list of dicts with 'start' and 'end'
        # in SAMPLES. We convert to seconds.
        timestamps = _svad.get_speech_timestamps(
            wav_tensor,
            model,
            sampling_rate=sr,
            threshold=cfg.threshold,
            min_speech_duration_ms=cfg.min_speech_ms,
            min_silence_duration_ms=cfg.min_silence_ms,
            speech_pad_ms=cfg.speech_pad_ms,
            return_seconds=True,  # silero-vad ≥ 4.0 supports this flag
        )

        segments: list[tuple[float, float]] = []
        for t in timestamps:
            start = float(t["start"]) if isinstance(t["start"], (int, float)) else float(t["start"])
            end   = float(t["end"])   if isinstance(t["end"],   (int, float)) else float(t["end"])
            segments.append((start, end))

        return segments
