"""FastAPI application factory — GPU edition.

Identical to app/main.py except:
  - Uses get_gpu_settings() and build_runtime_gpu()
  - Uses build_queue_gpu() and the GPU-aware recovery helpers

Run with:
    uvicorn gpu.main_gpu:factory --factory --reload

(or swap --reload for a port: --port 8001 to run alongside the CPU server)
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.observability.logging import configure_logging
from app.observability.middleware import RequestLogMiddleware
from app.pipeline.recovery import RecoveryConfig, RecoverySweeper, recover_runs
from gpu.config_gpu import GPUSettings, get_gpu_settings
from gpu.pipeline.queues_gpu import build_queue_gpu
from gpu.runtime_gpu import build_runtime_gpu


def create_app(settings: GPUSettings | None = None) -> FastAPI:
    settings = settings or get_gpu_settings()
    configure_logging(settings.log_level, settings.log_format)
    rt = build_runtime_gpu(settings)
    queue = build_queue_gpu(settings, rt)
    cfg = RecoveryConfig.from_settings(settings)
    sweeper = RecoverySweeper(
        lambda: recover_runs(rt.session_factory, queue, cfg),
        settings.recovery_interval_s,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        sweeper.start()
        yield
        sweeper.stop()
        queue.shutdown()
        rt.engine.dispose()

    app = FastAPI(title="ASR Testbed — GPU", lifespan=lifespan)
    app.state.settings = settings
    app.state.engine = rt.engine
    app.state.session_factory = rt.session_factory
    app.state.storage = rt.storage
    app.state.registry = rt.registry
    app.state.queue = queue
    app.add_middleware(RequestLogMiddleware)
    app.include_router(router)
    return app


def factory() -> FastAPI:
    """Run with: uvicorn gpu.main_gpu:factory --factory"""
    return create_app()
