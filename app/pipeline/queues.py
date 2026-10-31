import logging

from app.config import Settings

log = logging.getLogger("queues")


class CeleryQueue:
    """Same `enqueue(run_id)` / `shutdown()` interface as InProcessQueue."""

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
        return CeleryQueue()
    if settings.queue_backend == "inprocess":
        from app.pipeline.worker import InProcessQueue
        return InProcessQueue(runtime.processor, settings.worker_threads)
    raise ValueError(f"Unknown QUEUE_BACKEND: {settings.queue_backend}")
