from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import update
from sqlalchemy.orm import sessionmaker

from app.db.models import Base, RunStatus, TranscriptionRun
from app.db.repositories import AudioRepo, RunRepo, UserRepo
from app.db.session import make_engine
from app.pipeline.recovery import RecoveryConfig, recover_runs

NOW = datetime.now(timezone.utc)
CFG = RecoveryConfig(stuck_after_s=900, queued_after_s=900, max_attempts=3, batch=1000)


class FakeQueue:
    def __init__(self): self.ids = []
    def enqueue(self, run_id): self.ids.append(run_id)


@pytest.fixture
def sf(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path/'r.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def make_run(sf, **fields) -> str:
    with sf() as s:
        u = UserRepo(s).create(f"u{id(fields)}")[0]
        a = AudioRepo(s).add(user_id=u.id, storage_key=f"audio/{id(fields)}.wav", checksum="c",
                             duration=1, original_sample_rate=16000, size_bytes=1)
        r = RunRepo(s).add(a.id, "stub", {})
        if fields:
            s.execute(update(TranscriptionRun).where(TranscriptionRun.id == r.id).values(**fields))
        s.commit()
        return r.id


def status_of(sf, run_id):
    with sf() as s:
        r = s.get(TranscriptionRun, run_id)
        return r.status, r.error


def test_stuck_processing_is_requeued_and_enqueued(sf):
    rid = make_run(sf, status=RunStatus.processing, started_at=NOW - timedelta(seconds=1000), attempts=1)
    q = FakeQueue()
    rep = recover_runs(sf, q, CFG, now=NOW)
    assert rep.requeued_stuck == 1 and q.ids == [rid]
    assert status_of(sf, rid)[0] is RunStatus.queued


def test_healthy_long_running_task_is_left_alone(sf):
    rid = make_run(sf, status=RunStatus.processing, started_at=NOW - timedelta(seconds=100), attempts=1)
    q = FakeQueue()
    rep = recover_runs(sf, q, CFG, now=NOW)
    assert rep.requeued_stuck == 0 and q.ids == []
    assert status_of(sf, rid)[0] is RunStatus.processing


def test_poison_pill_is_failed_not_retried_forever(sf):
    rid = make_run(sf, status=RunStatus.processing, started_at=NOW - timedelta(seconds=1000), attempts=3)
    q = FakeQueue()
    rep = recover_runs(sf, q, CFG, now=NOW)
    status, err = status_of(sf, rid)
    assert rep.failed_exhausted == 1 and q.ids == []
    assert status is RunStatus.failed and "gave up after 3 attempts" in err


def test_orphaned_queued_run_is_reenqueued_once_per_window(sf):
    rid = make_run(sf, status=RunStatus.queued, enqueued_at=NOW - timedelta(seconds=1000))
    fresh = make_run(sf, status=RunStatus.queued, enqueued_at=NOW - timedelta(seconds=10))
    q = FakeQueue()
    assert recover_runs(sf, q, CFG, now=NOW).reenqueued_orphans == 1
    assert q.ids == [rid]                       # the fresh one is a healthy backlog item: untouched
    assert recover_runs(sf, q, CFG, now=NOW).reenqueued_orphans == 0   # immediate 2nd sweep: rate-limited
    assert q.ids == [rid]
    later = NOW + timedelta(seconds=1000)       # a window later it is retried again
    assert recover_runs(sf, q, CFG, now=later).reenqueued_orphans == 2
    assert fresh in q.ids


def test_done_and_failed_runs_are_never_touched(sf):
    a = make_run(sf, status=RunStatus.done, started_at=NOW - timedelta(seconds=5000))
    b = make_run(sf, status=RunStatus.failed, started_at=NOW - timedelta(seconds=5000))
    q = FakeQueue()
    assert recover_runs(sf, q, CFG, now=NOW) == type(recover_runs(sf, q, CFG, now=NOW))()
    assert status_of(sf, a)[0] is RunStatus.done and status_of(sf, b)[0] is RunStatus.failed


def test_claim_records_start_time_and_attempt(sf):
    rid = make_run(sf)
    with sf() as s:
        repo = RunRepo(s)
        assert repo.claim(rid)
        s.commit()
        r = s.get(TranscriptionRun, rid)
        assert r.attempts == 1 and r.started_at is not None
