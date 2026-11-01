"""Derived-artifact cache: resampled audio and chunk lists, stored under the `derived/` prefix.

KEY      derived/<checksum[:2]>/<checksum>/<kind>-<params_hash>.<ext>
         checksum    = SHA-256 of the ORIGINAL audio (so identical uploads share their artifacts)
         params_hash = first 16 hex chars of SHA-256 over the canonical JSON of every input that affects the
                       artifact, including the algorithm version constants below.

INVALIDATION RULE (there is no in-place invalidation)
  An artifact is a pure function of (original audio, parameters, algorithm version). Change any input and the key
  changes, so the old entry is simply never looked up again. Nothing is ever overwritten (blobs are immutable).
    * changed settings (sample rate, chunk length, VAD thresholds)   -> new key automatically
    * changed code that alters the output                            -> bump RESAMPLE_ALGO_VERSION / CHUNK_ALGO_VERSION
    * upgraded VAD library                                           -> the backend's `version` is part of the key
    * chunk lists also depend on the resampled audio, so they include RESAMPLE_ALGO_VERSION
  Entries live as long as their source audio: `prune_orphans` deletes derived data whose original audio is gone.

CONCURRENCY  Builds are deterministic and written atomically (temp file, then link). If two workers build the same
  artifact at once, the loser sees BlobExistsError and uses the winner's identical copy.
CORRUPTION   Chunk-list files record their own checksum and parameters; a mismatch is treated as a miss
  (the bad entry is deleted and rebuilt).
"""
import hashlib
import io
import json
import logging
import re
import tempfile
import time
from dataclasses import dataclass
from typing import BinaryIO, Protocol

import numpy as np
import soundfile as sf
import soxr

from app.observability.logging import log_event
from app.pipeline.run_config import ChunkingConfig, VadConfig
from app.storage.base import BlobExistsError, StorageBackend

log = logging.getLogger("derived")

RESAMPLE_ALGO_VERSION = 1  # soxr HQ streaming resampler, mono mean downmix, PCM_16 WAV
CHUNK_ALGO_VERSION = 1     # fixed windows with overlap / VAD segments with forced cuts, times rounded to ms

_CHECKSUM = re.compile(r"[0-9a-f]{64}")
_BLOCK_FRAMES = 1 << 16


class VadUnavailable(Exception):
    """The configured VAD backend is not installed/registered."""


class VadBackend(Protocol):
    name: str
    version: str  # part of the cache key: upgrading the library invalidates cached chunk lists

    def detect_speech_segments(self, audio: BinaryIO, cfg: VadConfig) -> list[tuple[float, float]]: ...


@dataclass(frozen=True)
class Chunk:
    start: float
    end: float


def params_hash(params: dict) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def derived_key(checksum: str, kind: str, params: dict, ext: str) -> str:
    if not _CHECKSUM.fullmatch(checksum):
        raise ValueError("checksum must be a 64-character lowercase hex SHA-256")
    return f"derived/{checksum[:2]}/{checksum}/{kind}-{params_hash(params)}.{ext}"


def plan_fixed_chunks(duration: float, max_chunk_s: float, overlap_s: float) -> list[Chunk]:
    step = max_chunk_s - overlap_s
    chunks, start = [], 0.0
    while start < duration:
        end = min(start + max_chunk_s, duration)
        chunks.append(Chunk(round(start, 3), round(end, 3)))
        if end >= duration:
            break
        start += step
    return chunks


def plan_vad_chunks(segments: list[tuple[float, float]], max_chunk_s: float) -> list[Chunk]:
    """One chunk per speech segment; a segment longer than max_chunk_s is force-cut every max_chunk_s."""
    chunks = []
    for start, end in sorted(segments):
        while end - start > max_chunk_s:
            chunks.append(Chunk(round(start, 3), round(start + max_chunk_s, 3)))
            start += max_chunk_s
        if end > start:
            chunks.append(Chunk(round(start, 3), round(end, 3)))
    return chunks


class DerivedCache:
    def __init__(self, storage: StorageBackend):
        self.storage = storage

    # ---------------------------------------------------------------- resampled audio
    def resampled_params(self, sample_rate: int) -> dict:
        return {"sample_rate": sample_rate, "channels": 1, "subtype": "PCM_16", "algo": RESAMPLE_ALGO_VERSION}

    def resampled(self, source_key: str, checksum: str, sample_rate: int) -> BinaryIO:
        """Open stream of the mono WAV at `sample_rate`, building it on first use. Caller closes it."""
        key = derived_key(checksum, "resampled", self.resampled_params(sample_rate), "wav")
        hit = self.storage.exists(key)
        if not hit:
            started = time.perf_counter()
            self._build_resampled(source_key, key, sample_rate)
            log_event(log, "derived.cache", kind="resampled", hit=False, key=key,
                      build_s=round(time.perf_counter() - started, 3))
        else:
            log_event(log, "derived.cache", kind="resampled", hit=True, key=key)
        return self.storage.get_audio(key)

    def _build_resampled(self, source_key: str, key: str, sample_rate: int) -> None:
        with self.storage.get_audio(source_key) as src_stream, sf.SoundFile(src_stream) as src, \
                tempfile.TemporaryFile(suffix=".wav") as tmp:
            stream = None
            if src.samplerate != sample_rate:
                stream = soxr.ResampleStream(src.samplerate, sample_rate, num_channels=1,
                                             dtype="float32", quality="HQ")
            with sf.SoundFile(tmp, "w", samplerate=sample_rate, channels=1, subtype="PCM_16", format="WAV") as out:
                for block in src.blocks(blocksize=_BLOCK_FRAMES, dtype="float32"):
                    mono = block.mean(axis=1) if block.ndim > 1 else block
                    out.write(stream.resample_chunk(mono.astype("float32", copy=False)) if stream else mono)
                if stream:
                    out.write(stream.resample_chunk(np.zeros(0, dtype="float32"), last=True))
            tmp.seek(0)
            try:
                self.storage.save_audio(key, tmp)
            except BlobExistsError:
                pass  # another worker built the identical artifact first

    # ---------------------------------------------------------------- chunk lists
    def chunks(self, source_key: str, checksum: str, chunking: ChunkingConfig, sample_rate: int,
               vad_backends: dict[str, VadBackend] | None = None) -> list[Chunk]:
        backend = None
        if chunking.strategy == "vad":
            backend = (vad_backends or {}).get(chunking.vad.backend)
            if backend is None:
                raise VadUnavailable(f"VAD backend '{chunking.vad.backend}' is not available")
        params = {"sample_rate": sample_rate, "chunking": chunking.model_dump(mode="json"),
                  "vad_version": backend.version if backend else None,
                  "algo": CHUNK_ALGO_VERSION, "resample_algo": RESAMPLE_ALGO_VERSION}
        key = derived_key(checksum, "chunks", params, "json")

        cached = self._read_chunks(key, checksum, params)
        if cached is not None:
            log_event(log, "derived.cache", kind="chunks", hit=True, key=key, n_chunks=len(cached))
            return cached

        started = time.perf_counter()
        with self.resampled(source_key, checksum, sample_rate) as wav:
            duration = sf.info(wav).duration
            if chunking.strategy == "fixed":
                chunks = plan_fixed_chunks(duration, chunking.max_chunk_s, chunking.overlap_s)
            else:
                wav.seek(0)
                chunks = plan_vad_chunks(backend.detect_speech_segments(wav, chunking.vad), chunking.max_chunk_s)
        payload = {"format": 1, "source_sha256": checksum, "params": params, "duration_s": round(duration, 3),
                   "chunks": [{"start": c.start, "end": c.end} for c in chunks]}
        try:
            self.storage.save_audio(key, io.BytesIO(json.dumps(payload, sort_keys=True).encode()))
        except BlobExistsError:
            pass
        log_event(log, "derived.cache", kind="chunks", hit=False, key=key, n_chunks=len(chunks),
                  build_s=round(time.perf_counter() - started, 3))
        return chunks

    def _read_chunks(self, key: str, checksum: str, params: dict) -> list[Chunk] | None:
        if not self.storage.exists(key):
            return None
        try:
            with self.storage.get_audio(key) as f:
                data = json.load(f)
            if data["source_sha256"] != checksum or data["params"] != params:
                raise ValueError("entry does not match its key")
            return [Chunk(c["start"], c["end"]) for c in data["chunks"]]
        except Exception:  # noqa: BLE001  corrupt or foreign entry: drop it and rebuild
            log_event(log, "derived.corrupt", level=logging.WARNING, key=key)
            self.storage.delete(key)
            return None


def prune_orphans(storage: StorageBackend, session_factory, dry_run: bool = False) -> list[str]:
    """Delete derived artifacts whose original audio no longer exists in the database."""
    from sqlalchemy import select

    from app.db.models import Audio

    by_checksum: dict[str, list[str]] = {}
    for key in storage.list_keys("derived/"):
        parts = key.split("/")
        if len(parts) == 4 and _CHECKSUM.fullmatch(parts[2]):
            by_checksum.setdefault(parts[2], []).append(key)
    checksums = list(by_checksum)
    live: set[str] = set()
    with session_factory() as s:
        for i in range(0, len(checksums), 500):
            live.update(s.scalars(select(Audio.checksum).where(Audio.checksum.in_(checksums[i:i + 500]))))
    removed = []
    for checksum, keys in by_checksum.items():
        if checksum not in live:
            for key in keys:
                if not dry_run:
                    storage.delete(key)
                removed.append(key)
    return removed
