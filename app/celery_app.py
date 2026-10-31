"""Celery app + the single task. Start a worker with:

    celery -A app.celery_app worker --concurrency=4 --loglevel=info

Reliability model:
  * acks_late + reject_on_worker_lost: a message is acknowledged only after the task finishes, so if a
    worker process dies mid-task the broker redelivers it.
  * Redelivery alone is not enough: a dead worker leaves its run in 'processing', so claim() refuses the
    redelivered message. app/pipeline/recovery.py resets such runs. Both layers are safe to overlap because
    claim() and complete() are atomic and first-writer-wins.
"""
import logging

from celery import Celery

from app.config import get_settings
from app.runtime import Runtime, build_runtime

log = logging.getLogger("celery_app")
settings = get_settings()

celery_app = Celery("asr", broker=settings.redis_url)  # no result backend: results live in our DB
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    task_default_queue="asr",
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # don't hoard messages: fair dispatch, less lost on a crash
    task_soft_time_limit=settings.task_soft_time_limit_s,
    task_time_limit=settings.task_hard_time_limit_s,
    broker_transport_options={"visibility_timeout": settings.visibility_timeout_s},
    broker_connection_retry_on_startup=True,
)

_runtime: Runtime | None = None


def get_runtime() -> Runtime:
    """Built lazily, once per worker *child* process (never share an engine across fork)."""
    global _runtime
    if _runtime is None:
        _runtime = build_runtime(settings, settings.worker_db_pool_size, settings.worker_db_max_overflow)
    return _runtime


@celery_app.task(name="asr.process_run", acks_late=True, reject_on_worker_lost=True)
def process_run(run_id: str) -> None:
    get_runtime().processor.process(run_id)
