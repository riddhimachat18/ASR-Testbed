"""Part H: every run records per-stage timings, cache hits and its runtime; rescoring keeps them; the
leaderboard never adds WERs scored under different normalizations."""
import json

import pytest

from app.db.models import TranscriptionRun
from app.evaluation.service import evaluate_run
from tests.conftest import make_wav
from tests.test_evaluation import _post, _wait_done

FIXED = json.dumps({"chunking": {"strategy": "fixed"}})  # no VAD needed in the light test environment


def _run(client, run_id) -> TranscriptionRun:
    with client.app.state.session_factory() as s:
        run = s.get(TranscriptionRun, run_id)
        s.expunge(run)
        return run


def test_done_run_has_stage_timings_cache_hits_and_runtime(client, make_user):
    _, key = make_user("alice")
    sub = _post(client, key, reference="this is a dummy transcript", model_version="stub-prepared", config=FIXED)
    assert _wait_done(client, key, sub["run_id"])["status"] == "done"
    p = _run(client, sub["run_id"]).profile
    t = p["timings"]
    assert set(t) == {"prepare_s", "inference_s", "score_s", "total_s"}
    assert t["inference_s"] >= 0.04  # the stub sleeps 0.05 s
    assert t["prepare_s"] + t["inference_s"] + t["score_s"] <= t["total_s"] + 1e-3
    assert p["cache_hits"] == {"chunks": False, "resampled": False} and p["n_chunks"] >= 1
    assert p["runtime"]["python"] and "libs" in p["runtime"] and p["runtime"]["vad_device"] == "cpu"


def test_second_run_on_the_same_audio_hits_the_cache(client, make_user):
    _, key = make_user("alice")
    first = _post(client, key, model_version="stub-prepared", config=FIXED)
    _wait_done(client, key, first["run_id"])
    second = client.post("/audio", headers={"X-API-Key": key},
                         data={"model_version": "stub-prepared", "config": FIXED},
                         files={"file": ("a.wav", make_wav(0.2), "audio/wav")}).json()  # same bytes, same checksum
    _wait_done(client, key, second["run_id"])
    assert _run(client, second["run_id"]).profile["cache_hits"] == {"chunks": True}


def test_failed_run_keeps_its_partial_profile(client, make_user):
    _, key = make_user("alice")
    sub = _post(client, key, config=json.dumps({"model_options": {"force_error": True}}))
    assert _wait_done(client, key, sub["run_id"])["status"] == "failed"
    p = _run(client, sub["run_id"]).profile
    assert "total_s" in p["timings"] and "inference_s" not in p["timings"]


def test_rescoring_changes_the_normalization_but_keeps_the_profile(client, make_user):
    _, key = make_user("alice")
    sub = _post(client, key, reference="THIS IS A DUMMY TRANSCRIPT")
    _wait_done(client, key, sub["run_id"])
    before = _run(client, sub["run_id"])
    assert before.metrics["normalization"] == "whisper-en-v1"
    with client.app.state.session_factory() as s:
        assert evaluate_run(s, sub["run_id"], "basic-v1")
        s.commit()
    after = _run(client, sub["run_id"])
    assert after.metrics["normalization"] == "basic-v1" and after.profile == before.profile


def test_leaderboard_keeps_normalizations_apart(client, make_user):
    _, key = make_user("alice")
    a = _post(client, key, reference="this is a dummy transcript", dataset="d1")
    b = _post(client, key, reference="this is a dummy transcript", dataset="d1")
    for sub in (a, b):
        _wait_done(client, key, sub["run_id"])
    with client.app.state.session_factory() as s:
        evaluate_run(s, b["run_id"], "basic-v1")
        s.commit()
    lb = client.get("/datasets/d1/leaderboard", headers={"X-API-Key": key}).json()
    assert sorted(r["normalization"] for r in lb) == ["basic-v1", "whisper-en-v1"]
    assert all(r["runs"] == 1 for r in lb)
