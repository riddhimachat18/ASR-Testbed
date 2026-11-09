"""RunProcessor and InProcessQueue — GPU edition.

Identical to app/pipeline/worker.py.
Copied here so the GPU pipeline is entirely self-contained and the two code
paths can evolve independently without risk of accidental cross-contamination.

The only runtime difference is that RunProcessor is wired to ModelRegistryGPU
and AudioPreparerGPU (via gpu/runtime_gpu.py), so every inference call goes
through the GPU-backed wrapper and GPU VAD backend.
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy.orm import Session, sessionmaker

from app.db.repositories import AudioRepo, RunRepo
from app.observability.logging import log_context, log_event
from app.evaluation.service import score_text
from gpu.pipeline.prepare_gpu import AudioPreparerGPU
from gpu.pipeline.registry_gpu import ModelRegistryGPU
from app.pipeline.run_config import RunConfig
from app.storage.base import StorageBackend

log = logging.getLogger("worker_gpu")


class RunProcessorGPU:
    """Processes one run on GPU. Idempotent: a run that isn't 'queued' is skipped."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        storage: StorageBackend,
        registry: ModelRegistryGPU,
        preparer: AudioPreparerGPU | None = None,
    ):
        self.session_factory = session_factory
        self.storage = storage
        self.registry = registry
        self.preparer = preparer or AudioPreparerGPU(storage)

    def process(self, run_id: str) -> None:
        with log_context(run_id=run_id):
            self._process(run_id)

    def _process(self, run_id: str) -> None:
        with self.session_factory() as s:
            runs = RunRepo(s)
            if not runs.claim(run_id):
                log_event(log, "run.skipped", reason="not claimable (already taken, finished or unknown)")
                s.rollback()
                return
            s.commit()
            start = time.perf_counter()
            try:
                run = runs.get(run_id)
                audio = AudioRepo(s).get(run.audio_id)
                wrapper = self.registry.get(run.model_version)
                cfg = RunConfig.model_validate(run.config or {})
                storage_key = audio.storage_key
                audio_id = audio.id
                checksum = audio.checksum
                model_version = run.model_version
                config_hash = run.config_hash
                attempt = run.attempts
                s.commit()
                log_event(
                    log, "run.started",
                    model_version=model_version,
                    config_hash=config_hash,
                    attempt=attempt,
                    audio_id=audio_id,
                )
                if wrapper.needs_preparation:
                    stream, prepared = self.preparer.prepare(storage_key, checksum, cfg)
                else:
                    stream, prepared = self.storage.get_audio(storage_key), None
                with stream:
                    text = wrapper.transcribe(stream, cfg, prepared)
                scores = self._score(s, audio_id, text)
                latency = time.perf_counter() - start
                applied = runs.complete(run_id, text, latency, scores)
                if applied:
                    log_event(
                        log, "run.done",
                        model_version=model_version,
                        latency_s=round(latency, 3),
                        wer=scores["wer"] if scores else None,
                        scored=scores is not None,
                    )
            except Exception as e:  # noqa: BLE001
                s.rollback()
                latency = time.perf_counter() - start
                log_event(
                    log, "run.failed",
                    level=logging.ERROR,
                    exc_info=True,
                    error=repr(e),
                    latency_s=round(latency, 3),
                )
                applied = runs.mark_failed(run_id, repr(e), latency)
            s.commit()
            if not applied:
                log_event(
                    log, "run.discarded",
                    level=logging.WARNING,
                    reason="state changed while processing; this result was discarded",
                )

    @staticmethod
    def _score(s: Session, audio_id: str, text: str) -> dict | None:
        """Best effort: a scoring problem must never stop the run from finishing."""
        try:
            return score_text(s, audio_id, text)
        except Exception:  # noqa: BLE001
            s.rollback()
            log_event(
                log, "run.scoring_failed",
                level=logging.ERROR,
                exc_info=True,
                audio_id=audio_id,
                note="the run is stored unscored; scripts.evaluate can retry",
            )
            return None


class InProcessQueueGPU:
    """Thread pool for GPU workers. Same enqueue/shutdown/stats interface as InProcessQueue."""

    def __init__(self, processor: RunProcessorGPU, threads: int):
        self.processor = processor
        self.pool = ThreadPoolExecutor(
            max_workers=threads, thread_name_prefix="asr-gpu-worker"
        )

    def enqueue(self, run_id: str) -> None:
        self.pool.submit(self.processor.process, run_id)

    def stats(self) -> dict:
        return {"inprocess_queue_pending": self.pool._work_queue.qsize()}

    def shutdown(self) -> None:
        self.pool.shutdown(wait=True)
