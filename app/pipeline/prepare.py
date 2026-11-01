from dataclasses import dataclass
from typing import BinaryIO

from app.pipeline.derived import Chunk, DerivedCache, VadBackend
from app.pipeline.run_config import RunConfig
from app.storage.base import StorageBackend


@dataclass(frozen=True)
class Prepared:
    sample_rate: int
    chunks: list[Chunk]


def _default_vad_backends() -> dict[str, VadBackend]:
    """Build the default VAD backend map. Silero is always available (installed as a dependency)."""
    from app.pipeline.silero_vad import SileroVadBackend
    silero = SileroVadBackend()
    return {silero.name: silero}


class AudioPreparer:
    """Turns an original audio blob into what a model consumes: resampled audio plus a chunk plan, both cached."""

    def __init__(self, storage: StorageBackend, vad_backends: dict[str, VadBackend] | None = None):
        self.cache = DerivedCache(storage)
        # Use caller-supplied backends, or default to Silero
        self.vad_backends = vad_backends if vad_backends is not None else _default_vad_backends()

    def prepare(self, source_key: str, checksum: str, cfg: RunConfig) -> tuple[BinaryIO, Prepared]:
        sr = cfg.audio.sample_rate
        chunks = self.cache.chunks(source_key, checksum, cfg.chunking, sr, self.vad_backends)
        return self.cache.resampled(source_key, checksum, sr), Prepared(sr, chunks)
