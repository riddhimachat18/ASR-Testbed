import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select, update

from app.config import Settings
from app.db.models import RunStatus, TranscriptionRun as Run
from app.observability.logging import log_event

log = logging.getLogger("recovery")


@dataclass
class RecoveryConfig:
    stuck_after_s: int
    queued_after_s: int
    max_attempts: int
    batch: int

    @classmethod
    def from_settings(cls, s: Settings) -> "RecoveryConfig":
        return cls(s.stuck_after_s, s.queued_after_s, s.max_attempts, s.recovery_batch)


@dataclass
class RecoveryReport:
    requeued_stuck: int = 0
    failed_exhausted: int = 0
    reenqueued_orphans: int = 0


def recover_runs(session_factory, queue, cfg: RecoveryConfig, now: datetime | None = None) -> RecoveryReport:
    """Safe to run from many processes at once: every change is a guarded UPDATE (first writer wins),
    and the broker is only touched after the database commit."""
    now = now or datetime.now(timezone.utc)
    stuck_cutoff = now - timedelta(seconds=cfg.stuck_after_s)
    queued_cutoff = now - timedelta(seconds=cfg.queued_after_s)
    report, to_enqueue = RecoveryReport(), []

    with session_factory() as s:
        # 1. Runs that already used all their attempts and are stuck again: stop retrying a poison pill.
        report.failed_exhausted = s.execute(
            update(Run)
            .where(Run.status == RunStatus.processing, Run.started_at < stuck_cutoff, Run.attempts >= cfg.max_attempts)
            .values(status=RunStatus.failed,
                    error=f"gave up after {cfg.max_attempts} attempts: worker died or timed out each time")
        ).rowcount

        # 2. Runs stuck in 'processing' (worker presumed dead): back to 'queued' and re-enqueue.
        stuck_ids = s.execute(
            update(Run)
            .where(Run.status == RunStatus.processing, Run.started_at < stuck_cutoff, Run.attempts < cfg.max_attempts)
            .values(status=RunStatus.queued, enqueued_at=now)
            .returning(Run.id)
        ).scalars().all()
        report.requeued_stuck = len(stuck_ids)
        to_enqueue += stuck_ids

        # 3. Orphans: 'queued' but not (re)enqueued recently, e.g. crash between commit and enqueue, or the
        #    broker lost the message. Bumping enqueued_at rate-limits this to once per queued_after_s per run,
        #    so a long but healthy backlog is not re-published over and over.
        orphan_candidates = s.scalars(
            select(Run.id)
            .where(Run.status == RunStatus.queued, func.coalesce(Run.enqueued_at, Run.created_at) < queued_cutoff)
            .order_by(Run.created_at).limit(cfg.batch)
        ).all()
        if orphan_candidates:
            claimed = s.execute(
                update(Run)
                .where(Run.id.in_(orphan_candidates), Run.status == RunStatus.queued,
                       func.coalesce(Run.enqueued_at, Run.created_at) < queued_cutoff)
                .values(enqueued_at=now)
                .returning(Run.id)
            ).scalars().all()
            report.reenqueued_orphans = len(claimed)
            to_enqueue += claimed
        s.commit()

    for run_id in to_enqueue:  # after commit
        queue.enqueue(run_id)
    if to_enqueue or report.failed_exhausted:
        log_event(log, "recovery.sweep", level=logging.WARNING, requeued_stuck=report.requeued_stuck,
                  failed_exhausted=report.failed_exhausted, reenqueued_orphans=report.reenqueued_orphans)
    return report


class RecoverySweeper:
    """Runs recover_runs once at startup and then every `interval_s` seconds (0 = disabled)."""

    def __init__(self, fn, interval_s: int):
        self.fn, self.interval_s = fn, interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.interval_s <= 0:
            return
        self._thread = threading.Thread(target=self._loop, name="recovery-sweeper", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.fn()
            except Exception:  # noqa: BLE001
                log.exception("recovery sweep failed")
            self._stop.wait(self.interval_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
