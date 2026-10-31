import time

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.db.models import Audio, Dataset, ReferenceTranscript, TranscriptionRun
from tests.conftest import make_wav

STUB_TEXT = "this is a dummy transcript"


def _post(client, key, reference=None, dataset=None, **extra):
    data = {k: v for k, v in {"reference": reference, "dataset": dataset, **extra}.items() if v is not None}
    r = client.post("/audio", headers={"X-API-Key": key}, data=data,
                    files={"file": ("a.wav", make_wav(0.2), "audio/wav")})
    assert r.status_code == 202, r.text
    return r.json()


def _wait_done(client, key, run_id):
    end = time.time() + 5
    while time.time() < end:
        r = client.get(f"/runs/{run_id}", headers={"X-API-Key": key}).json()
        if r["status"] in ("done", "failed"):
            return r
        time.sleep(0.02)
    raise AssertionError("timeout")


def test_run_is_scored_when_reference_uploaded_with_audio(client, make_user):
    _, key = make_user("alice")
    sub = _post(client, key, reference="This is a dummy transcript!", dataset="demo")
    run = _wait_done(client, key, sub["run_id"])
    assert run["wer"] == 0.0 and run["cer"] == 0.0     # formatting differences are normalised away
    res = client.get(f"/runs/{sub['run_id']}/result", headers={"X-API-Key": key}).json()
    assert res["wer"] == 0.0


def test_wer_value_end_to_end(client, make_user):
    _, key = make_user("alice")
    sub = _post(client, key, reference="this is a different transcript")   # 1 substitution of 5 words
    run = _wait_done(client, key, sub["run_id"])
    assert run["wer"] == pytest.approx(1 / 5)
    with client.app.state.session_factory() as s:
        m = s.get(TranscriptionRun, sub["run_id"]).metrics
        assert m["word_substitutions"] == 1 and m["ref_words"] == 5 and m["normalization"] == "basic-v1"


def test_run_without_reference_is_unscored(client, make_user):
    _, key = make_user("alice")
    run = _wait_done(client, key, _post(client, key)["run_id"])
    assert run["status"] == "done" and run["wer"] is None


def test_reference_added_later_rescores_finished_runs(client, make_user):
    _, key = make_user("alice")
    sub = _post(client, key)
    _wait_done(client, key, sub["run_id"])
    r = client.put(f"/audio/{sub['audio_id']}/reference", headers={"X-API-Key": key},
                   json={"text": "this is a dummy transcript", "language": "en", "source": "manual"})
    assert r.status_code == 200 and r.json()["runs_rescored"] == 1
    assert client.get(f"/runs/{sub['run_id']}", headers={"X-API-Key": key}).json()["wer"] == 0.0
    # changing the reference re-scores again
    client.put(f"/audio/{sub['audio_id']}/reference", headers={"X-API-Key": key}, json={"text": "totally different"})
    assert client.get(f"/runs/{sub['run_id']}", headers={"X-API-Key": key}).json()["wer"] == pytest.approx(5 / 2)


def test_empty_reference_is_recorded_not_scored(client, make_user):
    _, key = make_user("alice")
    sub = _post(client, key, reference="...")
    run = _wait_done(client, key, sub["run_id"])
    assert run["status"] == "done" and run["wer"] is None
    with client.app.state.session_factory() as s:
        assert "empty" in s.get(TranscriptionRun, sub["run_id"]).metrics["error"]


def test_leaderboard_is_micro_averaged(client, make_user):
    _, key = make_user("alice")
    # file 1: 5-word reference, 0 errors; file 2: 1-word reference, 4 insertions -> per-file WERs 0.0 and 4.0
    a = _post(client, key, reference="this is a dummy transcript", dataset="d1")
    b = _post(client, key, reference="this", dataset="d1")
    for sub in (a, b):
        _wait_done(client, key, sub["run_id"])
    lb = client.get("/datasets/d1/leaderboard", headers={"X-API-Key": key}).json()
    assert len(lb) == 1 and lb[0]["model_version"] == "stub" and lb[0]["runs"] == 2
    # micro average = (0 + 4 errors) / (5 + 1 words) = 0.667, NOT the mean of 0.0 and 4.0
    assert lb[0]["wer"] == pytest.approx(4 / 6)
    assert lb[0]["ref_words"] == 6 and lb[0]["audio_files"] == 2


def test_datasets_and_references_are_private_to_their_owner(client, make_user):
    _, k1 = make_user("alice")
    _, k2 = make_user("bob")
    sub = _post(client, k1, reference="x", dataset="mine")
    assert client.get("/datasets/mine/leaderboard", headers={"X-API-Key": k2}).status_code == 404
    assert client.put(f"/audio/{sub['audio_id']}/reference", headers={"X-API-Key": k2},
                      json={"text": "hijack"}).status_code == 404
    assert client.get("/datasets/nope/leaderboard", headers={"X-API-Key": k1}).status_code == 404


def test_same_dataset_name_is_reused_not_duplicated(client, make_user):
    _, key = make_user("alice")
    _post(client, key, dataset="d")
    _post(client, key, dataset="d")
    with client.app.state.session_factory() as s:
        assert len(s.scalars(select(Dataset)).all()) == 1
        assert {a.dataset_id for a in s.scalars(select(Audio)).all()} == {s.scalars(select(Dataset)).one().id}


def test_scoring_failure_never_undoes_a_finished_run(client, make_user, monkeypatch):
    import app.pipeline.worker as worker
    def boom(*a, **k): raise RuntimeError("scorer exploded")
    monkeypatch.setattr(worker, "evaluate_run", boom)
    _, key = make_user("alice")
    run = _wait_done(client, key, _post(client, key, reference="x")["run_id"])
    assert run["status"] == "done" and run["wer"] is None


def test_db_constraints(client, make_user):
    _, key = make_user("alice")
    sub = _post(client, key, reference="x")
    with client.app.state.session_factory() as s:
        s.add(ReferenceTranscript(audio_id=sub["audio_id"], text="second reference for the same audio"))
        with pytest.raises(IntegrityError):
            s.commit()
