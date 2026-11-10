"""What a run actually ran on, recorded in transcription_runs.profile["runtime"].

The config hash says what a run was asked to do; this says where it ran (device and precision actually used,
GPU name, library versions, peak memory), so a result can be traced to its hardware and software."""
from __future__ import annotations

import importlib.metadata
import platform
import sys
from functools import lru_cache

_LIBS = ("faster-whisper", "ctranslate2", "torch", "silero-vad", "jiwer", "numpy")


def _version(pkg: str) -> str | None:
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


@lru_cache(maxsize=1)
def _static() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.node(), "libs": {p: _version(p) for p in _LIBS}}


@lru_cache(maxsize=4)
def _gpu_name(device: str) -> str | None:
    if device != "cuda":
        return None
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001 - profiling must never fail a run
        return None


def peak_rss_mb() -> float | None:
    """Peak resident memory of this process in MB (None on Windows, which has no `resource` module)."""
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024, 1)  # bytes on macOS, KB on Linux


def runtime_info(wrapper, cfg, vad_device: str) -> dict:
    info = {**_static(), "vad_device": vad_device}
    resolve = getattr(wrapper, "resolve_hardware", None)
    if resolve is not None:
        device, compute_type = resolve(cfg)
        info.update(device=device, compute_type=compute_type, gpu=_gpu_name(device),
                    model_size=getattr(wrapper, "model_size", None))
    info["peak_rss_mb"] = peak_rss_mb()
    return info
