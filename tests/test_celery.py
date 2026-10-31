import time

import pytest

import app.celery_app as ca
from app.config import Settings
from app.db.models import Base, RunStatus
from app.pipeline.queues import CeleryQueue, build_queue
from app.runtime import build_runtime
from tests.conftest import make_wav


@pytest.fixture
def eager_runtime(settings, monkeypatch):
    """Run Celery tasks inline (no broker) against a test DB."""
    rt = build_runtime(settings)
    Base.metadata.create_all(rt.engine)
    monkeypatch.setattr(ca, "_runtime", rt)
    monkeypatch.setitem(ca.celery_app.conf, "task_always_eager", True)
    monkeypatch.setitem(ca.celery_app.conf, "task_eager_propagates", True)
    return rt


def _submit_via_api(settings, monkeypatch):
    from fastapi.testclient import TestClient
    from app.db.repositories import UserRepo
    from app.main import create_app
    s2 = Settings(**{**settings.model_dump(), "queue_backend": "celery"})
    app = create_app(s2)
    return app, TestClient(app)


def test_celery_backend_processes_run_end_to_end(settings, eager_runtime):
    from fastapi.testclient import TestClient
    from app.db.repositories import UserRepo
    from app.main import create_app
    s2 = Settings(**{**settings.model_dump(), "queue_backend": "celery"})
    app = create_app(s2)
    assert isinstance(app.state.queue, CeleryQueue)
    with TestClient(app) as c:
        with app.state.session_factory() as s:
            _, key = UserRepo(s).create("alice"); s.commit()
        r = c.post("/audio", headers={"X-API-Key": key}, files={"file": ("a.wav", make_wav(), "audio/wav")})
        assert r.status_code == 202
        run = c.get(f"/runs/{r.json()['run_id']}", headers={"X-API-Key": key}).json()
        assert run["status"] == "done"   # eager mode ran the task inline


def test_duplicate_delivery_is_harmless(eager_runtime):
    from app.db.repositories import AudioRepo, RunRepo, UserRepo
    import io
    rt = eager_runtime
    with rt.session_factory() as s:
        u, _ = UserRepo(s).create("bob")
        rt.storage.save_audio("audio/x/a.wav", io.BytesIO(make_wav()))
        a = AudioRepo(s).add(user_id=u.id, storage_key="audio/x/a.wav", checksum="c", duration=1,
                             original_sample_rate=44100, size_bytes=1)
        rid = RunRepo(s).add(a.id, "stub", {}).id
        s.commit()
    ca.process_run.delay(rid)
    ca.process_run.delay(rid)   # broker redelivers the same message
    with rt.session_factory() as s:
        run = RunRepo(s).get(rid)
        assert run.status is RunStatus.done and run.attempts == 1   # second delivery was a no-op


def test_enqueue_survives_broker_outage(monkeypatch):
    def boom(*a, **k): raise ConnectionError("redis down")
    monkeypatch.setattr(ca.process_run, "delay", boom)
    CeleryQueue().enqueue("some-run-id")   # must not raise: the row stays 'queued' for the sweep


def test_unknown_backend_rejected(settings):
    with pytest.raises(ValueError):
        build_queue(Settings(**{**settings.model_dump(), "queue_backend": "nope"}), None)


def test_timing_invariants_enforced():
    with pytest.raises(ValueError):
        Settings(task_soft_time_limit_s=600, task_hard_margin_s=60, stuck_after_s=300)
