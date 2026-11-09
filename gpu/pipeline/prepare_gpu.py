"""AudioPreparer — GPU edition.

Identical to app/pipeline/prepare.py except:
  - The default VAD backend is SileroVadBackendGPU instead of SileroVadBackend,
    so VAD inference runs on the same CUDA device as Whisper.
  - The device string is forwarded from GPUSettings.

If you want VAD on CPU and Whisper on GPU, instantiate AudioPreparer directly
with the CPU SileroVadBackend — this class is just the GPU-default convenience.
"""
from dataclasses import dataclass
from typing import BinaryIO

from app.pipeline.derived import Chunk, DerivedCache, VadBackend
from app.pipeline.run_config import RunConfig
from app.storage.base import StorageBackend


@dataclass(frozen=True)
class Prepared:
    sample_rate: int
    chunks: list[Chunk]


def _default_vad_backends_gpu(device: str = "cuda") -> dict[str, VadBackend]:
    """Build the default VAD backend map using the GPU-backed Silero backend."""
    from gpu.pipeline.silero_vad_gpu import SileroVadBackendGPU
    silero = SileroVadBackendGPU(device=device)
    return {silero.name: silero}


class AudioPreparerGPU:
    """Turns an original audio blob into what a model consumes: resampled audio + chunk plan.

    Same as AudioPreparer (CPU) but defaults to the GPU Silero-VAD backend.
    The DerivedCache layer (resampling, storage, chunk JSON) is shared with the
    CPU path — derived artifacts are device-agnostic so they can be reused across runs.
    """

    def __init__(
        self,
        storage: StorageBackend,
        vad_backends: dict[str, VadBackend] | None = None,
        device: str = "cuda",
    ):
        self.cache = DerivedCache(storage)
        self.vad_backends = (
            vad_backends
            if vad_backends is not None
            else _default_vad_backends_gpu(device)
        )

    def prepare(
        self, source_key: str, checksum: str, cfg: RunConfig
    ) -> tuple[BinaryIO, Prepared]:
        sr = cfg.audio.sample_rate
        chunks = self.cache.chunks(source_key, checksum, cfg.chunking, sr, self.vad_backends)
        return self.cache.resampled(source_key, checksum, sr), Prepared(sr, chunks)
