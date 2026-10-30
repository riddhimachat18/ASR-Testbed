from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import Settings, get_settings
from app.db.session import make_engine, make_session_factory
from app.pipeline.registry import ModelRegistry
from app.pipeline.worker import InProcessQueue, RunProcessor
from app.storage.factory import get_storage


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    engine = make_engine(settings.database_url, settings.db_pool_size,
                         settings.db_max_overflow, settings.db_pool_timeout)
    session_factory = make_session_factory(engine)
    storage = get_storage(settings)
    registry = ModelRegistry(settings)
    queue = InProcessQueue(RunProcessor(session_factory, storage, registry), settings.worker_threads)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        queue.shutdown()
        engine.dispose()

    app = FastAPI(title="ASR Testbed", lifespan=lifespan)
    app.state.settings, app.state.engine = settings, engine
    app.state.session_factory, app.state.storage = session_factory, storage
    app.state.registry, app.state.queue = registry, queue
    app.include_router(router)
    return app


def factory() -> FastAPI:
    """Run with: uvicorn app.main:factory --factory"""
    return create_app()
