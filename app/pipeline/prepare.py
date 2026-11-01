from dataclasses import dataclass
from typing import BinaryIO

from app.pipeline.derived import Chunk, DerivedCache, VadBackend
from app.pipeline.run_config import RunConfig
from app.storage.base import StorageBackend


@dataclass(frozen=True)
class Prepared:
    sample_rate: int
    chunks: list[Chunk]


class AudioPreparer:
    """Turns an original audio blob into what a model consumes: resampled audio plus a chunk plan, both cached."""

    def __init__(self, storage: StorageBackend, vad_backends: dict[str, VadBackend] | None = None):
        self.cache = DerivedCache(storage)
        self.vad_backends = vad_backends or {}

    def prepare(self, source_key: str, checksum: str, cfg: RunConfig) -> tuple[BinaryIO, Prepared]:
        sr = cfg.audio.sample_rate
        chunks = self.cache.chunks(source_key, checksum, cfg.chunking, sr, self.vad_backends)
        return self.cache.resampled(source_key, checksum, sr), Prepared(sr, chunks)
