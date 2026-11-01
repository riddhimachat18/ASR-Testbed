import io
import json
import logging
import re
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import update

from app.config import Settings
from app.db.models import Base, TranscriptionRun
from app.db.repositories import AudioRepo, RunRepo, UserRepo
from app.main import create_app
from app.observability.logging import (JsonFormatter, configure_logging, current_context, log_context,
                                       log_event)
from app.observability.metrics import _esc
from tests.conftest import make_wav


def _logger_with_buffer(formatter=None):
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(formatter or JsonFormatter())
    lg = logging.getLogger(f"test.{id(buf)}")
    lg.handlers, lg.propagate = [handler], False
    lg.setLevel(logging.DEBUG)
    return lg, buf


@pytest.fixture(autouse=True)
def _factory():
    configure_logging("INFO", "json")      # installs the record factory (idempotent)


# ------------------------------------------------------------------ JSON logging
def test_json_line_has_core_fields_event_fields_and_context():
    lg, buf = _logger_with_buffer()
    with log_context(run_id="run123", request_id="req9"):
        log_event(lg, "run.done", latency_s=1.5, wer=0.1, weird=object)
    rec = json.loads(buf.getvalue())
    assert rec["msg"] == "run.done" and rec["level"] == "INFO" and rec["logger"] == lg.name
    assert rec["run_id"] == "run123" and rec["request_id"] == "req9"
    assert rec["latency_s"] == 1.5 and rec["wer"] == 0.1 and isinstance(rec["weird"], str)  # unserialisable -> str
    assert rec["ts"].endswith("Z") and isinstance(rec["pid"], int)


def test_exceptions_are_included():
    lg, buf = _logger_with_buffer()
    try:
        1 / 0
    except ZeroDivisionError:
        log_event(lg, "run.failed", level=logging.ERROR, exc_info=True, error="boom")
    rec = json.loads(buf.getvalue())
    assert rec["level"] == "ERROR" and "ZeroDivisionError" in rec["exc"] and rec["error"] == "boom"


def test_context_is_scoped_nested_and_not_inherited_by_other_threads():
    lg, buf = _logger_with_buffer()
    assert current_context() == {}
    with log_context(run_id="a"):
        with log_context(request_id="r"):
            assert current_context() == {"run_id": "a", "request_id": "r"}
        assert current_context() == {"run_id": "a"}
        t = threading.Thread(target=lambda: log_event(lg, "from.thread"))
        t.start(); t.join()                                        # a pool thread must not leak/borrow a run_id
    log_event(lg, "after")
    lines = [json.loads(x) for x in buf.getvalue().splitlines()]
    assert all("run_id" not in x for x in lines) and current_context() == {}


def test_configure_logging_is_idempotent_and_switches_format(capsys):
    configure_logging("INFO", "json"); configure_logging("INFO", "json")
    assert sum(getattr(h, "_asr_handler", False) for h in logging.getLogger().handlers) == 1
    logging.getLogger("cfg.test").info("hello json")
    assert json.loads(capsys.readouterr().err.strip().splitlines()[-1])["msg"] == "hello json"
    configure_logging("INFO", "text")
    with log_context(run_id="r7"):
        log_event(logging.getLogger("cfg.test"), "run.done", wer=0.2)
    line = capsys.readouterr().err.strip().splitlines()[-1]
    assert "run.done" in line and "run_id=r7" in line and "wer=0.2" in line
    configure_logging("INFO", "json")


# ------------------------------------------------------------------ events through the real app
def _post(client, key, **data):
    return client.post("/audio", headers={"X-API-Key": key}, data=data,
                       files={"file": ("a.wav", make_wav(0.2), "audio/wav")})


def _wait_done(client, key, run_id):
    end = time.time() + 5
    while time.time() < end:
        r = client.get(f"/runs/{run_id}", headers={"X-API-Key": key}).json()
        if r["status"] in ("done", "failed"):
            return r
        time.sleep(0.02)
    raise AssertionError("timeout")


def _events(caplog, name):
    return [r for r in caplog.records if r.getMessage() == name]


def test_worker_lifecycle_events_carry_the_run_id(client, make_user, caplog):
    caplog.set_level(logging.INFO)
    _, key = make_user("alice")
    sub = _post(client, key, reference="this is a dummy transcript").json()
    _wait_done(client, key, sub["run_id"])
    started, done = _events(caplog, "run.started"), _events(caplog, "run.done")
    assert len(started) == 1 and len(done) == 1
    assert started[0].run_id == done[0].run_id == sub["run_id"]       # attached by log_context, not passed by hand
    assert done[0].fields["wer"] == 0.0 and done[0].fields["latency_s"] >= 0 and done[0].fields["scored"] is True
    assert started[0].fields["attempt"] == 1 and len(started[0].fields["config_hash"]) == 64


def test_submit_is_logged_with_run_id_and_request_id(client, make_user, caplog):
    caplog.set_level(logging.INFO)
    _, key = make_user("alice")
    sub = _post(client, key, dataset="d1").json()
    ev = _events(caplog, "run.submitted")[0]
    assert ev.fields["run_id"] == sub["run_id"] and ev.fields["dataset"] == "d1" and ev.fields["size_bytes"] > 0
    assert len(ev.request_id) == 16                                    # same request context as the access line


def test_failed_run_logs_an_error_event_with_the_traceback(client, make_user, caplog):
    caplog.set_level(logging.INFO)
    _, key = make_user("alice")
    sub = _post(client, key, config='{"model_options": {"force_error": true}}').json()
    _wait_done(client, key, sub["run_id"])
    ev = _events(caplog, "run.failed")[0]
    assert ev.levelno == logging.ERROR and ev.run_id == sub["run_id"] and ev.exc_info and "forced stub" in ev.fields["error"]


# ------------------------------------------------------------------ request middleware
def test_every_response_gets_a_request_id_and_one_access_event(client, make_user, caplog):
    caplog.set_level(logging.INFO)
    _, key = make_user("alice")
    run_id = _post(client, key).json()["run_id"]
    r1 = client.get(f"/runs/{run_id}", headers={"X-API-Key": key})
    r2 = client.get(f"/runs/{run_id}", headers={"X-API-Key": key})
    assert re.fullmatch(r"[0-9a-f]{16}", r1.headers["x-request-id"]) and r1.headers["x-request-id"] != r2.headers["x-request-id"]
    access = [r for r in _events(caplog, "http.request") if r.fields["path"] == "/runs/{run_id}"]
    assert len(access) == 2
    f = access[0].fields
    assert f["method"] == "GET" and f["status"] == 200 and f["run_id"] == run_id and f["duration_ms"] >= 0
    assert access[0].request_id == r1.headers["x-request-id"]


def test_error_statuses_are_logged_and_probes_are_quiet(client, caplog):
    caplog.set_level(logging.INFO)
    assert client.get("/runs/x").status_code == 401
    client.get("/healthz"); client.get("/metrics")
    paths = [(r.fields["path"], r.fields["status"]) for r in _events(caplog, "http.request")]
    assert ("/runs/{run_id}", 401) in paths
    assert not any(p in ("/healthz", "/metrics") for p, _ in paths)   # DEBUG level: no scraper noise


# ------------------------------------------------------------------ /metrics
SAMPLE = re.compile(r'^([a-z_]+)(?:\{(.*)\})? (-?[0-9.e+]+)$')


def parse_metrics(text):
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line:
            continue
        m = SAMPLE.match(line)
        assert m, f"not valid exposition format: {line!r}"
        labels = tuple(sorted(re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', m.group(2) or "")))
        out[(m.group(1), labels)] = float(m.group(3))
    return out


def test_metrics_report_status_counts_in_prometheus_format(client, make_user):
    _, key = make_user("alice")
    ids = [_post(client, key).json()["run_id"] for _ in range(3)]
    bad = _post(client, key, config='{"model_options": {"force_error": true}}').json()["run_id"]
    for rid in ids + [bad]:
        _wait_done(client, key, rid)
    r = client.get("/metrics")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain; version=0.0.4")
    m = parse_metrics(r.text)
    assert m[("asr_runs", (("status", "done"),))] == 3
    assert m[("asr_runs", (("status", "failed"),))] == 1
    assert m[("asr_runs", (("status", "queued"),))] == 0 and m[("asr_runs", (("status", "processing"),))] == 0  # zero-filled
    assert m[("asr_runs_by_model", (("model_version", "stub"), ("status", "done")))] == 3
    assert m[("asr_inprocess_queue_pending", ())] == 0
    for name in ("asr_runs", "asr_oldest_queued_age_seconds", "asr_inprocess_queue_pending"):
        assert f"# HELP {name} " in r.text and f"# TYPE {name} gauge" in r.text


def test_oldest_queued_age_shows_a_stuck_queue(client, make_user):
    _, key = make_user("alice")
    assert parse_metrics(client.get("/metrics").text)[("asr_oldest_queued_age_seconds", ())] == 0   # nothing queued
    with client.app.state.session_factory() as s:                   # a run that nobody will ever pick up
        u = UserRepo(s).create("stuck")[0]
        a = AudioRepo(s).add(user_id=u.id, storage_key="audio/stuck.wav", checksum="c", duration=1,
                             original_sample_rate=8000, size_bytes=1)
        rid = RunRepo(s).add(a.id, "stub", {}).id
        s.execute(update(TranscriptionRun).where(TranscriptionRun.id == rid)
                  .values(created_at=datetime.now(timezone.utc) - timedelta(seconds=120)))
        s.commit()
    m = parse_metrics(client.get("/metrics").text)
    assert 119 <= m[("asr_oldest_queued_age_seconds", ())] < 180
    assert m[("asr_runs", (("status", "queued"),))] == 1


def test_metrics_token_protects_the_endpoint(settings):
    app = create_app(Settings(**{**settings.model_dump(), "metrics_token": "s3cret"}))
    Base.metadata.create_all(app.state.engine)
    with TestClient(app) as c:
        assert c.get("/metrics").status_code == 401
        assert c.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert c.get("/metrics", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert c.get("/healthz").status_code == 200                    # probes stay open


def test_broker_down_is_reported_not_fatal(settings):
    s2 = Settings(**{**settings.model_dump(), "queue_backend": "celery", "redis_url": "redis://127.0.0.1:1/0"})
    app = create_app(s2)
    Base.metadata.create_all(app.state.engine)
    with TestClient(app) as c:
        started = time.time()
        r = c.get("/metrics")
        assert r.status_code == 200 and time.time() - started < 5      # short timeouts: a dead Redis cannot hang scrapes
        m = parse_metrics(r.text)
        assert m[("asr_broker_up", ())] == 0 and ("asr_broker_queue_length", ()) not in m
        assert m[("asr_runs", (("status", "queued"),))] == 0           # database metrics still served


def test_label_values_are_escaped():
    assert _esc('a"b\\c\nd') == 'a\\"b\\\\c\\nd'
