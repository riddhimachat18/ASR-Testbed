import logging
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy.orm import Session, sessionmaker

from app.db.repositories import AudioFeaturesRepo, AudioRepo, RunRepo, StageEventRepo
from app.observability.logging import log_context, log_event
from app.observability.tracing import Trace, activate, span
from app.evaluation.metrics import DEFAULT_NORMALIZATION, get_normalizer
from app.evaluation.service import score_text
from app.pipeline.audio_features import FEATURES_VERSION, features_from_wav
from app.pipeline.prepare import AudioPreparer
from app.pipeline.registry import ModelRegistry
from app.pipeline.run_config import RunConfig
from app.pipeline.runtime_info import runtime_info
from app.storage.base import StorageBackend

log = logging.getLogger("worker")


class RunProcessor:
    """Processes one run. Idempotent: a run that isn't `queued` is skipped.

    Every finished or failed run gets a `profile`: per-stage timings (prepare = fetch + resample + VAD or cache
    read; inference; scoring; total), cache hits, and the runtime it ran on (runtime_info.py).

    Every attempt is also traced (app/observability/tracing.py): queue wait, each cache lookup, VAD, every chunk's
    slice / features / encoder / decoder call, scoring and the database write become rows in stage_events,
    written right after the run's final status is committed (telemetry can fail without failing the run). The run's audio features are computed the first
    time the file is processed, after the latency is measured, so they never count against it."""

    def __init__(self, session_factory: sessionmaker[Session], storage: StorageBackend, registry: ModelRegistry,
                 preparer: AudioPreparer | None = None, normalization: str = DEFAULT_NORMALIZATION,
                 vad_device: str = "cpu"):
        self.session_factory, self.storage, self.registry = session_factory, storage, registry
        self.preparer = preparer or AudioPreparer(storage, vad_device=vad_device)
        get_normalizer(normalization)  # fail at startup, not on the first run, if NORMALIZER is misspelled
        self.normalization, self.vad_device = normalization, vad_device

    def process(self, run_id: str) -> None:
        with log_context(run_id=run_id):  # every log line below, from any module, carries run_id
            self._process(run_id)

    def _process(self, run_id: str) -> None:
        with self.session_factory() as s:
            runs = RunRepo(s)
            if not runs.claim(run_id):
                log_event(log, "run.skipped", reason="not claimable (already taken, finished or unknown)")
                s.rollback()
                return
            s.commit()
            run = runs.get(run_id)
            trace = Trace(run_id, attempt=run.attempts or 1)
            if run.started_at is not None:
                queued_at = run.enqueued_at or run.created_at
                if queued_at is not None:
                    started_at, queued_at = _utc(run.started_at), _utc(queued_at)
                    trace.add("queue_wait", queued_at, (started_at - queued_at).total_seconds())
            with activate(trace):
                self._execute(s, runs, run_id, trace)

    def _execute(self, s: Session, runs: RunRepo, run_id: str, trace: Trace) -> None:
        start = time.perf_counter()
        timings: dict[str, float] = {}
        profile: dict = {"timings": timings}
        prepared, cfg, storage_key, checksum, audio_id = None, None, None, None, None
        with trace.span("run") as run_attrs:
            try:
                with span("db.load"):
                    run = runs.get(run_id)
                    audio = AudioRepo(s).get(run.audio_id)
                    wrapper = self.registry.get(run.model_version)
                    cfg = RunConfig.model_validate(run.config or {})  # {} (legacy rows) = all defaults
                    storage_key, audio_id, checksum = audio.storage_key, audio.id, audio.checksum
                    model_version, config_hash, attempt = run.model_version, run.config_hash, run.attempts
                    s.commit()  # release the DB connection while the (slow) model runs
                run_attrs.update(model_version=model_version, config_hash=config_hash, audio_id=audio_id)
                log_event(log, "run.started", model_version=model_version, config_hash=config_hash,
                          attempt=attempt, audio_id=audio_id)
                profile["runtime"] = runtime_info(wrapper, cfg, self.vad_device)
                t = time.perf_counter()
                with span("prepare"):
                    if wrapper.needs_preparation:  # cached resample + chunk plan (see derived.py)
                        stream, prepared = self.preparer.prepare(storage_key, checksum, cfg)
                        profile["cache_hits"] = prepared.cache_hits
                        profile["n_chunks"] = len(prepared.chunks)
                    else:
                        stream, prepared = self.storage.get_audio(storage_key), None
                timings["prepare_s"] = round(time.perf_counter() - t, 4)
                with stream:
                    t = time.perf_counter()
                    with span("transcribe"):
                        text = wrapper.transcribe(stream, cfg, prepared)
                    timings["inference_s"] = round(time.perf_counter() - t, 4)
                if getattr(wrapper, "last_stats", None) is not None:
                    profile["decoder"] = wrapper.last_stats  # segments, fallbacks, longest repeated word
                t = time.perf_counter()
                with span("score"):
                    scores = self._score(s, audio_id, text)
                timings["score_s"] = round(time.perf_counter() - t, 4)
                latency = time.perf_counter() - start
                timings["total_s"] = round(latency, 4)
                profile["stages"] = trace.summary()
                with span("db.write"):
                    applied = runs.complete(run_id, text, latency, scores, profile=profile)
                if applied:
                    log_event(log, "run.done", model_version=model_version, latency_s=round(latency, 3),
                              wer=scores["wer"] if scores else None, scored=scores is not None)
            except Exception as e:  # noqa: BLE001
                s.rollback()
                latency = time.perf_counter() - start
                log_event(log, "run.failed", level=logging.ERROR, exc_info=True, error=repr(e),
                          latency_s=round(latency, 3))
                timings["total_s"] = round(latency, 4)
                run_attrs["error"] = type(e).__name__
                profile["stages"] = trace.summary()
                applied = runs.mark_failed(run_id, repr(e), latency, profile=profile)
        s.commit()  # the run's outcome first: it must never depend on its telemetry
        try:
            StageEventRepo(s).add_all(trace.rows())
            s.commit()
        except Exception:  # noqa: BLE001
            s.rollback()
            log_event(log, "trace.write_failed", level=logging.WARNING, exc_info=True)
        if not applied:
            log_event(log, "run.discarded", level=logging.WARNING,
                      reason="state changed while processing; this result was discarded")
        if applied and prepared is not None and audio_id is not None:
            self._ensure_audio_features(s, audio_id, storage_key, checksum, cfg, prepared)

    def _ensure_audio_features(self, s: Session, audio_id: str, storage_key: str, checksum: str,
                               cfg: RunConfig, prepared) -> None:
        """Compute the file's audio features once (best effort; outside the measured latency)."""
        try:
            repo = AudioFeaturesRepo(s)
            existing = repo.get(audio_id)
            if existing is not None and existing.version == FEATURES_VERSION:
                return
            t = time.perf_counter()
            with self.preparer.cache.resampled(storage_key, checksum, prepared.sample_rate) as wav:
                feats = features_from_wav(wav, [(c.start, c.end) for c in prepared.chunks],
                                          vad=cfg.chunking.strategy == "vad")
            feats["extra"]["compute_ms"] = round((time.perf_counter() - t) * 1000.0, 2)
            repo.upsert(audio_id, feats)
            s.commit()
        except Exception:  # noqa: BLE001
            s.rollback()
            log_event(log, "audio_features.failed", level=logging.WARNING, exc_info=True, audio_id=audio_id)

    def _score(self, s: Session, audio_id: str, text: str) -> dict | None:
        """Best effort: a scoring problem must never stop the run from finishing.
        (scripts.evaluate can score it later.)"""
        try:
            return score_text(s, audio_id, text, self.normalization)
        except Exception:  # noqa: BLE001
            s.rollback()
            log_event(log, "run.scoring_failed", level=logging.ERROR, exc_info=True, audio_id=audio_id,
                      note="the run is stored unscored; scripts.evaluate can retry")
            return None


def _utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts


class InProcessQueue:
    """Thread pool for now. Same `enqueue(run_id)` interface a Celery/Redis queue will implement."""

    def __init__(self, processor: RunProcessor, threads: int):
        self.processor = processor
        self.pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="asr-worker")

    def enqueue(self, run_id: str) -> None:
        self.pool.submit(self.processor.process, run_id)

    def stats(self) -> dict:
        return {"inprocess_queue_pending": self.pool._work_queue.qsize()}

    def shutdown(self) -> None:
        self.pool.shutdown(wait=True)
