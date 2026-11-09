"""Queue builder — GPU edition.

Identical to app/pipeline/queues.py except:
  - Uses GPUSettings
  - InProcessQueue uses InProcessQueueGPU
  - CeleryQueue routes to the GPU Celery task (gpu/celery_app_gpu.py)

The queue backend selection logic (inprocess vs celery) is unchanged.
"""
import logging

from gpu.config_gpu import GPUSettings

log = logging.getLogger("queues_gpu")


class CeleryQueueGPU:
    """Same enqueue/shutdown/stats interface as InProcessQueueGPU but dispatches to Celery."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url

    def stats(self) -> dict:
        try:
            import redis
            r = redis.Redis.from_url(
                self.redis_url, socket_connect_timeout=0.5, socket_timeout=0.5
            )
            return {"broker_up": 1, "broker_queue_length": r.llen("asr")}
        except Exception:  # noqa: BLE001
            return {"broker_up": 0}

    def enqueue(self, run_id: str) -> None:
        from gpu.celery_app_gpu import process_run_gpu  # lazy import

        try:
            process_run_gpu.delay(run_id)
        except Exception:  # noqa: BLE001
            log.exception("could not enqueue run %s (GPU); recovery sweep will retry", run_id)

    def shutdown(self) -> None:
        pass


def build_queue_gpu(settings: GPUSettings, runtime):
    if settings.queue_backend == "celery":
        return CeleryQueueGPU(settings.redis_url)
    if settings.queue_backend == "inprocess":
        from gpu.pipeline.worker_gpu import InProcessQueueGPU
        return InProcessQueueGPU(runtime.processor, settings.worker_threads)
    raise ValueError(f"Unknown QUEUE_BACKEND: {settings.queue_backend}")
