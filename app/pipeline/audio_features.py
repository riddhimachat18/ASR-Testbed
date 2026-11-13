"""Descriptive audio features, computed once per audio file and stored in audio_features.

What they are for: the dashboard plots WER and latency against them (does accuracy drop with noise? does
latency follow chunk count rather than duration?), and later they are cheap, reference-free drift signals.

All are computed from the cached 16 kHz mono WAV and the run's chunk plan, with numpy only:
  rms_dbfs / peak_dbfs   overall level and peak, in dB relative to full scale (0 dBFS = the loudest possible)
  clip_ratio             share of samples at (or within 0.1% of) full scale: a clipped, distorted recording
  speech_ratio           share of the file inside VAD speech segments (None for fixed chunking: no VAD)
  snr_db                 rough signal-to-noise ratio: speech-segment level minus non-speech level. An estimate,
                         not a measurement: it needs at least 0.2 s of non-speech, otherwise None
  spectral_centroid_hz   "brightness": energy-weighted mean frequency over 25 ms frames. Muffled or
                         band-limited audio (phone line, distant mic) has a low centroid
  n_chunks               chunks in the plan. Whisper pads every chunk to a 30 s window, so
  padding_ratio          = 30 s x n_chunks / duration is how much more audio the encoder processes than exists
FEATURES_VERSION is stored with each row; bump it when a definition changes and re-run
scripts/extract_features.py --recompute.
"""
from __future__ import annotations

import math
from typing import BinaryIO

import numpy as np
import soundfile as sf

FEATURES_VERSION = 1
WHISPER_WINDOW_S = 30.0
_EPS = 1e-10


def _dbfs(x: np.ndarray) -> float | None:
    if x.size == 0:
        return None
    return round(20.0 * math.log10(float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) + _EPS), 2)


def _spectral_centroid(x: np.ndarray, sr: int, frame: int = 400, hop: int = 160) -> float | None:
    if x.size < frame:
        return None
    n = 1 + (x.size - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx] * np.hanning(frame)[None, :]
    mag = np.abs(np.fft.rfft(frames, axis=1))
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    energy = mag.sum(axis=1)
    if not energy.size or energy.sum() <= 0:
        return None
    keep = energy > 0
    cent = (mag[keep] @ freqs) / energy[keep]
    return round(float(np.average(cent, weights=energy[keep])), 1)  # loud frames (speech) dominate


def compute_features(samples: np.ndarray, sr: int, chunks: list[tuple[float, float]] | None,
                     vad: bool) -> dict:
    """samples: mono float32 in [-1, 1]; chunks: the run's chunk plan as (start_s, end_s)."""
    duration = samples.size / sr if sr else 0.0
    out: dict = {"version": FEATURES_VERSION, "duration_s": round(duration, 3),
                 "rms_dbfs": _dbfs(samples),
                 "peak_dbfs": round(20.0 * math.log10(float(np.max(np.abs(samples))) + _EPS), 2) if samples.size else None,
                 "clip_ratio": round(float(np.mean(np.abs(samples) >= 0.999)), 6) if samples.size else None,
                 "spectral_centroid_hz": _spectral_centroid(samples, sr),
                 "speech_ratio": None, "n_speech_segments": None, "snr_db": None,
                 "n_chunks": len(chunks) if chunks is not None else None, "padding_ratio": None, "extra": {}}
    if chunks and duration > 0:
        out["padding_ratio"] = round(WHISPER_WINDOW_S * len(chunks) / duration, 3)
        out["extra"]["chunk_s_mean"] = round(sum(e - s for s, e in chunks) / len(chunks), 3)
    if vad and chunks is not None and samples.size:
        mask = np.zeros(samples.size, dtype=bool)
        segments = 0
        last_end = None
        for s, e in sorted(chunks):
            a, b = max(int(s * sr), 0), min(int(e * sr), samples.size)
            mask[a:b] = True
            if last_end is None or s - last_end > 1e-3:  # force-cut pieces of one segment count once
                segments += 1
            last_end = e
        out["speech_ratio"] = round(float(mask.mean()), 4)
        out["n_speech_segments"] = segments
        speech, noise = samples[mask], samples[~mask]
        if speech.size and noise.size >= 0.2 * sr:
            s_db, n_db = _dbfs(speech), _dbfs(noise)
            out["snr_db"] = round(s_db - n_db, 2)
            out["extra"]["noise_floor_dbfs"] = n_db
    return out


def features_from_wav(wav: BinaryIO, chunks: list[tuple[float, float]] | None, vad: bool) -> dict:
    wav.seek(0)
    samples, sr = sf.read(wav, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return compute_features(samples, sr, chunks, vad)
