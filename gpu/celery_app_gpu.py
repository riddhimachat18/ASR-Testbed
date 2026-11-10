"""Celery app — GPU edition.

Identical to app/celery_app.py except:
  - Uses get_gpu_settings() and build_runtime_gpu()
  - The task name is "asr.process_run_gpu" (distinct from the CPU "asr.process_run")
    so CPU and GPU Celery workers can share the same Redis broker without
    accidentally stealing each other's tasks.

Start a GPU worker with:
    celery -A gpu.celery_app_gpu worker --pool=solo --concurrency=1 --loglevel=info

Note: --pool=solo is required on Windows (prefork pool has initialisation issues on Windows).
On Linux/macOS use the default prefork pool (omit --pool=solo).
"""
import logging

from celery import Celery
from celery.signals import setup_logging

from gpu.config_gpu import get_gpu_settings
from app.observability.logging import configure_logging
from gpu.runtime_gpu import RuntimeGPU, build_runtime_gpu

log = logging.getLogger("celery_app_gpu")
settings = get_gpu_settings()

celery_app_gpu = Celery("asr_gpu", broker=settings.redis_url)
celery_app_gpu.conf.update(
    task_serializer="json",
    accept_content=["json"],
    task_default_queue="asr",          # same queue as CPU workers — separated by task name
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_soft_time_limit=settings.task_soft_time_limit_s,
    task_time_limit=settings.task_hard_time_limit_s,
    broker_transport_options={
        "visibility_timeout": settings.visibility_timeout_s,
        "socket_timeout": 30,
        "socket_connect_timeout": 30,
        "redis_backend_use_ssl": False,
    },
    broker_connection_retry=True,
    broker_connection_retry_on_startup=True,
)


@setup_logging.connect
def _configure_worker_logging(**_):
    configure_logging(settings.log_level, settings.log_format)


_runtime: RuntimeGPU | None = None


def get_runtime() -> RuntimeGPU:
    """Built lazily, once per worker child process."""
    global _runtime
    if _runtime is None:
        _runtime = build_runtime_gpu(
            settings,
            settings.worker_db_pool_size,
            settings.worker_db_max_overflow,
        )
    return _runtime


@celery_app_gpu.task(name="asr.process_run_gpu", acks_late=True, reject_on_worker_lost=True)
def process_run_gpu(run_id: str) -> None:
    get_runtime().processor.process(run_id)
