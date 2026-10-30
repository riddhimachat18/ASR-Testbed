import time

from tests.conftest import make_wav


def _post(client, key, data=None, name="a.wav", **form):
    return client.post("/audio", headers={"X-API-Key": key},
                       files={"file": (name, data if data is not None else make_wav(), "audio/wav")}, data=form)


def _wait(client, key, run_id, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        r = client.get(f"/runs/{run_id}", headers={"X-API-Key": key}).json()
        if r["status"] in ("done", "failed"):
            return r
        time.sleep(0.02)
    raise AssertionError("timed out")


def test_full_lifecycle(client, make_user):
    _, key = make_user("alice")
    r = _post(client, key)
    assert r.status_code == 202
    run_id = r.json()["run_id"]
    assert _wait(client, key, run_id)["status"] == "done"
    res = client.get(f"/runs/{run_id}/result", headers={"X-API-Key": key})
    assert res.status_code == 200 and res.json()["transcript"] == "this is a dummy transcript"


def test_original_audio_stored_as_received(client, make_user):
    _, key = make_user("alice")
    _post(client, key, make_wav(1.0, rate=44100))
    with client.app.state.session_factory() as s:
        from app.db.models import Audio
        a = s.query(Audio).one()
        assert a.original_sample_rate == 44100 and abs(a.duration - 1.0) < 0.01
        assert a.storage_key.startswith("audio/") and not a.storage_key.startswith("/")
        assert len(a.checksum) == 64


def test_auth_required(client):
    assert client.post("/audio", files={"file": ("a.wav", make_wav())}).status_code == 401
    assert client.get("/runs/x", headers={"X-API-Key": "bad"}).status_code == 401


def test_users_cannot_see_each_others_runs(client, make_user):
    _, k1 = make_user("alice")
    _, k2 = make_user("bob")
    run_id = _post(client, k1).json()["run_id"]
    assert client.get(f"/runs/{run_id}", headers={"X-API-Key": k2}).status_code == 404


def test_result_not_ready_is_409(client, make_user, settings):
    client.app.state.registry._factories["stub"] = lambda: __import__(
        "app.pipeline.model_wrapper", fromlist=["x"]).StubModelWrapper(0.5)
    _, key = make_user("alice")
    run_id = _post(client, key).json()["run_id"]
    assert client.get(f"/runs/{run_id}/result", headers={"X-API-Key": key}).status_code == 409
    _wait(client, key, run_id)


def test_failure_path(client, make_user):
    _, key = make_user("alice")
    run_id = _post(client, key, config='{"force_error": true}').json()["run_id"]
    r = _wait(client, key, run_id)
    assert r["status"] == "failed" and "forced stub failure" in r["error"]


def test_rejects_bad_inputs(client, make_user):
    _, key = make_user("alice")
    assert _post(client, key, name="a.txt").status_code == 415
    assert _post(client, key, data=b"not audio at all").status_code == 422
    assert _post(client, key, model_version="nope").status_code == 422
    assert _post(client, key, config="[1]").status_code == 422
    assert _post(client, key, data=make_wav(60, rate=44100)).status_code == 413  # > 5MB test limit
    assert list(client.app.state.storage.list_keys()) == []  # failed uploads leave no blobs


def test_many_concurrent_submissions(client, make_user):
    from concurrent.futures import ThreadPoolExecutor
    _, key = make_user("alice")
    wav = make_wav(0.2)
    with ThreadPoolExecutor(8) as ex:
        ids = list(ex.map(lambda _: _post(client, key, wav).json()["run_id"], range(40)))
    assert len(set(ids)) == 40
    assert all(_wait(client, key, i, 20)["status"] == "done" for i in ids)
