"""Silero-VAD backend — GPU edition.

Identical to app/pipeline/silero_vad.py with one addition:
  the model and inference tensor are moved to the device specified in GPUSettings
  (device="cuda" by default).

Using the GPU for VAD is optional — VAD is fast enough on CPU even when Whisper
runs on GPU.  Having it here means both VAD *and* Whisper run on the same device,
which avoids unnecessary CPU↔GPU tensor copies and keeps the entire pipeline on GPU.

The public interface is the same as SileroVadBackend so AudioPreparer / DerivedCache
need no changes.
"""
from __future__ import annotations

import logging
import threading
from typing import BinaryIO

import silero_vad as _svad
import importlib.metadata

from app.pipeline.run_config import VadConfig

log = logging.getLogger("silero_vad_gpu")

# ── version string used as part of the derived-cache key ─────────────────────
try:
    _VERSION = importlib.metadata.version("silero-vad")
except importlib.metadata.PackageNotFoundError:
    _VERSION = "unknown"

# ── process-level singleton ───────────────────────────────────────────────────
_lock: threading.Lock = threading.Lock()
_model = None          # type: ignore[assignment]
_model_device: str = ""  # tracks which device the singleton was loaded on


def _get_model(device: str = "cuda"):
    """Load the Silero-VAD model once per process and place it on `device`.

    silero-vad ≥ 6.x: load_silero_vad() returns the model directly.
    silero-vad ≤ 5.x: load_silero_vad() returned (model, utils) tuple.
    Both forms are handled.
    """
    global _model, _model_device
    if _model is None or _model_device != device:
        with _lock:
            if _model is None or _model_device != device:
                log.info(
                    "loading silero-vad model (first call in this process)",
                    extra={"device": device},
                )
                result = _svad.load_silero_vad()
                m = result[0] if isinstance(result, tuple) else result
                try:
                    m = m.to(device)
                except Exception:
                    # Silero-VAD may not expose .to() on all versions; fall back gracefully.
                    log.warning(
                        "silero-vad model does not support .to('%s'); running on default device", device
                    )
                _model = m
                _model_device = device
    return _model


class SileroVadBackendGPU:
    """Implements app.pipeline.derived.VadBackend using Silero-VAD on GPU.

    Drop-in replacement for SileroVadBackend: same name/version attributes,
    same detect_speech_segments() signature.
    """

    name: str = "silero"
    version: str = _VERSION

    def __init__(self, device: str = "cuda"):
        self._device = device

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

        model = _get_model(self._device)

        data, sr = sf.read(audio, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Move the tensor to the same device as the model so inference stays on GPU.
        wav_tensor = torch.from_numpy(data).to(self._device)

        timestamps = _svad.get_speech_timestamps(
            wav_tensor,
            model,
            sampling_rate=sr,
            threshold=cfg.threshold,
            min_speech_duration_ms=cfg.min_speech_ms,
            min_silence_duration_ms=cfg.min_silence_ms,
            speech_pad_ms=cfg.speech_pad_ms,
            return_seconds=True,
        )

        segments: list[tuple[float, float]] = []
        for t in timestamps:
            start = float(t["start"])
            end = float(t["end"])
            segments.append((start, end))

        return segments
