from dataclasses import dataclass, field
from typing import BinaryIO

from app.pipeline.derived import Chunk, DerivedCache, VadBackend
from app.pipeline.run_config import RunConfig
from app.storage.base import StorageBackend


@dataclass(frozen=True)
class Prepared:
    sample_rate: int
    chunks: list[Chunk]
    cache_hits: dict[str, bool] = field(default_factory=dict)  # {"resampled": True, "chunks": False} (profiling)


def _default_vad_backends(device: str = "cpu") -> dict[str, VadBackend]:
    """Build the default VAD backend map: Silero if the silero-vad package is installed (it is in requirements.txt;
    a light test environment without torch gets an empty map, and a VAD run then fails with 'not available')."""
    import importlib.util

    if importlib.util.find_spec("silero_vad") is None:
        return {}
    from app.pipeline.silero_vad import SileroVadBackend
    silero = SileroVadBackend(device=device)
    return {silero.name: silero}


class AudioPreparer:
    """Turns an original audio blob into what a model consumes: resampled audio plus a chunk plan, both cached."""

    def __init__(self, storage: StorageBackend, vad_backends: dict[str, VadBackend] | None = None,
                 vad_device: str = "cpu"):
        self.cache = DerivedCache(storage)
        # Use caller-supplied backends, or default to Silero on vad_device
        self.vad_backends = vad_backends if vad_backends is not None else _default_vad_backends(vad_device)

    def prepare(self, source_key: str, checksum: str, cfg: RunConfig) -> tuple[BinaryIO, Prepared]:
        sr = cfg.audio.sample_rate
        hits: dict[str, bool] = {}
        chunks = self.cache.chunks(source_key, checksum, cfg.chunking, sr, self.vad_backends, hits=hits)
        stream = self.cache.resampled(source_key, checksum, sr)
        return stream, Prepared(sr, chunks, hits)
