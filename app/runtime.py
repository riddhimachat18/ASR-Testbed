from dataclasses import dataclass

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.db.session import make_engine, make_session_factory
from app.pipeline.prepare import AudioPreparer
from app.pipeline.registry import ModelRegistry
from app.pipeline.worker import RunProcessor
from app.storage.base import StorageBackend
from app.storage.factory import get_storage


@dataclass
class Runtime:
    settings: Settings
    engine: Engine
    session_factory: sessionmaker[Session]
    storage: StorageBackend
    registry: ModelRegistry
    processor: RunProcessor


def build_runtime(settings: Settings, pool_size: int | None = None, max_overflow: int | None = None) -> Runtime:
    """One place that wires DB + storage + models. Used by the API and by Celery workers."""
    engine = make_engine(
        settings.database_url,
        settings.db_pool_size if pool_size is None else pool_size,
        settings.db_max_overflow if max_overflow is None else max_overflow,
        settings.db_pool_timeout,
    )
    sf = make_session_factory(engine)
    storage = get_storage(settings)
    registry = ModelRegistry(settings)
    return Runtime(settings, engine, sf, storage, registry, RunProcessor(sf, storage, registry, AudioPreparer(storage)))
