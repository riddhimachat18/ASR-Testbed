"""Runtime builder — GPU edition.

Identical to app/runtime.py except:
  - Uses GPUSettings instead of Settings
  - Uses ModelRegistryGPU instead of ModelRegistry
  - Uses RunProcessorGPU and AudioPreparerGPU instead of their CPU counterparts

All DB, storage, and recovery logic is shared with the CPU path unchanged.
"""
from dataclasses import dataclass

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import make_engine, make_session_factory
from app.storage.base import StorageBackend
from app.storage.factory import get_storage
from gpu.config_gpu import GPUSettings
from gpu.pipeline.prepare_gpu import AudioPreparerGPU
from gpu.pipeline.registry_gpu import ModelRegistryGPU
from gpu.pipeline.worker_gpu import RunProcessorGPU


@dataclass
class RuntimeGPU:
    settings: GPUSettings
    engine: Engine
    session_factory: sessionmaker[Session]
    storage: StorageBackend
    registry: ModelRegistryGPU
    processor: RunProcessorGPU


def build_runtime_gpu(
    settings: GPUSettings,
    pool_size: int | None = None,
    max_overflow: int | None = None,
) -> RuntimeGPU:
    """One place that wires DB + storage + GPU models.
    Used by the GPU API server and by GPU Celery workers."""
    engine = make_engine(
        settings.database_url,
        settings.db_pool_size if pool_size is None else pool_size,
        settings.db_max_overflow if max_overflow is None else max_overflow,
        settings.db_pool_timeout,
    )
    sf = make_session_factory(engine)
    storage = get_storage(settings)
    registry = ModelRegistryGPU(settings)
    preparer = AudioPreparerGPU(storage, device=settings.device)
    processor = RunProcessorGPU(sf, storage, registry, preparer)
    return RuntimeGPU(settings, engine, sf, storage, registry, processor)
