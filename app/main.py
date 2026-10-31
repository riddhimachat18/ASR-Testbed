from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import Settings, get_settings
from app.pipeline.queues import build_queue
from app.pipeline.recovery import RecoveryConfig, RecoverySweeper, recover_runs
from app.runtime import build_runtime


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    rt = build_runtime(settings)
    queue = build_queue(settings, rt)
    cfg = RecoveryConfig.from_settings(settings)
    sweeper = RecoverySweeper(lambda: recover_runs(rt.session_factory, queue, cfg), settings.recovery_interval_s)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        sweeper.start()  # first sweep happens immediately at startup
        yield
        sweeper.stop()
        queue.shutdown()
        rt.engine.dispose()

    app = FastAPI(title="ASR Testbed", lifespan=lifespan)
    app.state.settings, app.state.engine = settings, rt.engine
    app.state.session_factory, app.state.storage = rt.session_factory, rt.storage
    app.state.registry, app.state.queue = rt.registry, queue
    app.include_router(router)
    return app


def factory() -> FastAPI:
    """Run with: uvicorn app.main:factory --factory"""
    return create_app()
