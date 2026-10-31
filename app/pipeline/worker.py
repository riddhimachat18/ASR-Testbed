import logging
import time
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy.orm import Session, sessionmaker

from app.db.repositories import AudioRepo, RunRepo
from app.evaluation.service import score_text
from app.pipeline.registry import ModelRegistry
from app.pipeline.run_config import RunConfig
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
                cfg = RunConfig.model_validate(run.config or {})  # {} (legacy rows) = all defaults
                storage_key, audio_id = audio.storage_key, audio.id
                s.commit()  # release the DB connection while the (slow) model runs
                with self.storage.get_audio(storage_key) as stream:
                    text = wrapper.transcribe(stream, cfg)
                scores = self._score(s, audio_id, text)
                applied = runs.complete(run_id, text, time.perf_counter() - start, scores)
            except Exception as e:  # noqa: BLE001
                s.rollback()
                log.exception("run %s failed", run_id)
                applied = runs.mark_failed(run_id, repr(e), time.perf_counter() - start)
            s.commit()
            if not applied:
                log.warning("run %s changed state while processing; this result was discarded", run_id)

    @staticmethod
    def _score(s: Session, audio_id: str, text: str) -> dict | None:
        """Best effort: a scoring problem must never stop the run from finishing.
        (scripts.evaluate can score it later.)"""
        try:
            return score_text(s, audio_id, text)
        except Exception:  # noqa: BLE001
            s.rollback()
            log.exception("scoring failed for audio %s; the run is stored unscored", audio_id)
            return None


class InProcessQueue:
    """Thread pool for now. Same `enqueue(run_id)` interface a Celery/Redis queue will implement."""

    def __init__(self, processor: RunProcessor, threads: int):
        self.processor = processor
        self.pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="asr-worker")

    def enqueue(self, run_id: str) -> None:
        self.pool.submit(self.processor.process, run_id)

    def shutdown(self) -> None:
        self.pool.shutdown(wait=True)
