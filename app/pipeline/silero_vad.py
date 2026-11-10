"""Silero-VAD backend for the derived-cache VadBackend protocol.

* Lazy imports. silero_vad and torch are imported on first use, not at module import, so the API process (which
  never runs VAD) starts without torch. Before Part H, importing the app required torch.
* Process-level model singleton, loaded on the first detect_speech_segments() call (about 1 s).
* Device. VAD runs on `device` (VAD_DEVICE, default "cpu"). If the model cannot be moved to the requested device
  we raise instead of silently staying on CPU, so a run never records a device it did not use.
* Version. The installed silero-vad version is read from package metadata and is part of the chunk-cache key, so a
  library upgrade invalidates cached chunk lists.
"""
from __future__ import annotations

import importlib.metadata
import logging
import threading
from typing import BinaryIO

from app.pipeline.run_config import VadConfig

log = logging.getLogger("silero_vad")

try:
    _VERSION = importlib.metadata.version("silero-vad")
except importlib.metadata.PackageNotFoundError:
    _VERSION = "unknown"

_lock = threading.Lock()
_models: dict[str, object] = {}  # device -> model


def _get_model(device: str):
    """Load Silero-VAD once per process and device. silero-vad 5.x returned (model, utils); 6.x the model."""
    if device not in _models:
        with _lock:
            if device not in _models:
                import silero_vad as svad

                log.info("loading silero-vad model", extra={"device": device})
                result = svad.load_silero_vad()
                model = result[0] if isinstance(result, tuple) else result
                if device != "cpu":
                    model = model.to(device)  # raises if unsupported: never fall back silently
                _models[device] = model
    return _models[device]


class SileroVadBackend:
    """Implements app.pipeline.derived.VadBackend using Silero-VAD."""

    name: str = "silero"
    version: str = _VERSION

    def __init__(self, device: str = "cpu"):
        self.device = device

    def detect_speech_segments(self, audio: BinaryIO, cfg: VadConfig) -> list[tuple[float, float]]:
        """(start_s, end_s) speech segments of a seekable 16 kHz mono WAV stream (not closed here)."""
        import silero_vad as svad
        import soundfile as sf
        import torch

        model = _get_model(self.device)
        data, sr = sf.read(audio, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        wav = torch.from_numpy(data).to(self.device)
        timestamps = svad.get_speech_timestamps(
            wav, model, sampling_rate=sr,
            threshold=cfg.threshold,
            min_speech_duration_ms=cfg.min_speech_ms,
            min_silence_duration_ms=cfg.min_silence_ms,
            speech_pad_ms=cfg.speech_pad_ms,
            return_seconds=True,
        )
        return [(float(t["start"]), float(t["end"])) for t in timestamps]
