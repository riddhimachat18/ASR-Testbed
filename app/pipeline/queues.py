import logging

from app.config import Settings

log = logging.getLogger("queues")


class CeleryQueue:
    """Same `enqueue(run_id)` / `shutdown()` / `stats()` interface as InProcessQueue."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url

    def stats(self) -> dict:
        """Messages waiting in Redis. Compare with queued runs in the database: a big gap means lost or duplicated
        messages (the recovery sweep's job)."""
        try:
            import redis
            r = redis.Redis.from_url(self.redis_url, socket_connect_timeout=0.5, socket_timeout=0.5)
            return {"broker_up": 1, "broker_queue_length": r.llen("asr")}
        except Exception:  # noqa: BLE001
            return {"broker_up": 0}

    def enqueue(self, run_id: str) -> None:
        from app.celery_app import process_run  # lazy: keeps the in-process path free of celery

        try:
            process_run.delay(run_id)
        except Exception:  # noqa: BLE001
            # The run row is already committed as 'queued'. Don't fail the upload because the broker is
            # briefly down: the recovery sweep re-enqueues it once the broker is back.
            log.exception("could not enqueue run %s; recovery sweep will retry", run_id)

    def shutdown(self) -> None:
        pass


def build_queue(settings: Settings, runtime):
    if settings.queue_backend == "celery":
        return CeleryQueue(settings.redis_url)
    if settings.queue_backend == "inprocess":
        from app.pipeline.worker import InProcessQueue
        return InProcessQueue(runtime.processor, settings.worker_threads)
    raise ValueError(f"Unknown QUEUE_BACKEND: {settings.queue_backend}")
