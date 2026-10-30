import logging
import time
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy.orm import Session, sessionmaker

from app.db.repositories import AudioRepo, RunRepo
from app.pipeline.registry import ModelRegistry
from app.storage.base import StorageBackend

log = logging.getLogger("worker")


class RunProcessor:
    """Processes one run. Idempotent: a run that isn't `queued` is skipped."""

    def __init__(self, session_factory: sessionmaker[Session], storage: StorageBackend, registry: ModelRegistry):
        self.session_factory, self.storage, self.registry = session_factory, storage, registry

    def process(self, run_id: str) -> None:
        with self.session_factory() as s:
            runs = RunRepo(s)
            if not runs.claim(run_id):
                log.info("run %s not claimable, skipping", run_id)
                s.rollback()
                return
            s.commit()
            start = time.perf_counter()
            try:
                run = runs.get(run_id)
                audio = AudioRepo(s).get(run.audio_id)
                wrapper = self.registry.get(run.model_version)
                storage_key, cfg = audio.storage_key, run.config or {}
                s.commit()  # release the DB connection while the (slow) model runs
                with self.storage.get_audio(storage_key) as stream:
                    text = wrapper.transcribe(stream, cfg)
                applied = runs.complete(run_id, text, time.perf_counter() - start)
            except Exception as e:  # noqa: BLE001
                s.rollback()
                log.exception("run %s failed", run_id)
                applied = runs.mark_failed(run_id, repr(e), time.perf_counter() - start)
            s.commit()
            if not applied:
                log.warning("run %s changed state while processing; this result was discarded", run_id)

class InProcessQueue:
    """Thread pool for now. Same `enqueue(run_id)` interface a Celery/Redis queue will implement."""

    def __init__(self, processor: RunProcessor, threads: int):
        self.processor = processor
        self.pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="asr-worker")

    def enqueue(self, run_id: str) -> None:
        self.pool.submit(self.processor.process, run_id)

    def shutdown(self) -> None:
        self.pool.shutdown(wait=True)
