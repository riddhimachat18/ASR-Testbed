import json
import time

import pytest

from app.config import Settings
from app.db.models import Audio, ReferenceTranscript, RunStatus, TranscriptionRun
from app.pipeline.model_wrapper import StubOptions
from app.pipeline.run_config import (ConfigInvalid, NoOptions, RunConfig, config_hash,
                                     resolve_run_config)
from tests.conftest import make_wav


def resolve(raw, model="stub", opts=StubOptions):
    return resolve_run_config(model, raw, opts)


# ---------------- resolution and hashing ----------------
def test_defaults_are_filled_in_completely():
    resolved, _ = resolve({})
    assert resolved["chunking"]["strategy"] == "vad"
    assert resolved["chunking"]["vad"]["backend"] == "silero"      # VAD block materialised for vad chunking
    assert resolved["decoding"]["beam_size"] == 5 and resolved["audio"]["sample_rate"] == 16000
    assert resolved["model_options"] == {"force_error": False} and resolved["schema_version"] == 1


def test_omitted_and_explicit_defaults_hash_the_same():
    _, h1 = resolve({})
    _, h2 = resolve({"decoding": {"beam_size": 5}, "chunking": {"strategy": "vad", "max_chunk_s": 30}})
    _, h3 = resolve({"decoding": {"beam_size": 5.0}})   # int/float spelling cannot matter either
    assert h1 == h2 == h3


def test_any_real_change_changes_the_hash():
    base = resolve({})[1]
    assert resolve({"decoding": {"beam_size": 1}})[1] != base
    assert resolve({"chunking": {"max_chunk_s": 20}})[1] != base
    assert resolve({"chunking": {"vad": {"threshold": 0.6}}})[1] != base
    assert resolve({"model_options": {"force_error": True}})[1] != base
    assert config_hash("whisper-small", resolve({})[0]) != config_hash("stub", resolve({})[0])  # model counts


def test_golden_hash_only_changes_when_you_mean_it():
    """If this fails you changed a default, a field, or the canonicalisation. That silently splits every
    leaderboard row from older runs. Update the constant only deliberately (and bump schema_version)."""
    assert resolve({})[1] == "37ffc2f867748bda61b825b16163e1c112ccb288e3f38ad14fdcbfbf7a6323d7"


# ---------------- validation ----------------
@pytest.mark.parametrize("raw,where", [
    ({"decoding": {"beem_size": 3}}, ["decoding", "beem_size"]),                  # typo is rejected, not ignored
    ({"decodng": {}}, ["decodng"]),
    ({"decoding": {"beam_size": 0}}, ["decoding", "beam_size"]),
    ({"decoding": {"beam_size": "five"}}, ["decoding", "beam_size"]),
    ({"decoding": {"language": "English"}}, ["decoding", "language"]),
    ({"decoding": {"temperature": 2}}, ["decoding", "temperature"]),
    ({"chunking": {"max_chunk_s": 120}}, ["chunking", "max_chunk_s"]),
    ({"chunking": {"strategy": "bogus"}}, ["chunking", "strategy"]),
    ({"audio": {"sample_rate": 44100}}, ["audio", "sample_rate"]),
    ({"chunking": {"vad": {"threshold": 1.5}}}, ["chunking", "vad", "threshold"]),
    ({"schema_version": 2}, ["schema_version"]),
])
def test_invalid_config_is_rejected_with_a_precise_location(raw, where):
    with pytest.raises(ConfigInvalid) as e:
        resolve(raw)
    assert any(err["loc"][:len(where)] == where for err in e.value.errors), e.value.errors


@pytest.mark.parametrize("raw,fragment", [
    ({"chunking": {"strategy": "vad", "overlap_s": 2}}, "overlap_s applies only"),
    ({"chunking": {"strategy": "fixed", "vad": {}}}, "vad settings apply only"),
    ({"chunking": {"strategy": "fixed", "max_chunk_s": 30, "overlap_s": 15}}, "less than half"),
])
def test_contradictory_combinations_are_rejected(raw, fragment):
    with pytest.raises(ConfigInvalid) as e:
        resolve(raw)
    assert fragment in json.dumps(e.value.errors)


def test_valid_fixed_window_config():
    resolved, _ = resolve({"chunking": {"strategy": "fixed", "max_chunk_s": 30, "overlap_s": 2}})
    assert resolved["chunking"]["vad"] is None and resolved["chunking"]["overlap_s"] == 2.0


def test_model_options_are_validated_per_model():
    with pytest.raises(ConfigInvalid) as e:
        resolve({"model_options": {"compute_type": "int8"}})      # stub declares no such option
    assert e.value.errors[0]["loc"][:2] == ["model_options", "compute_type"]
    with pytest.raises(ConfigInvalid):
        resolve({"model_options": {"anything": 1}}, opts=NoOptions)   # models without options accept none


def test_runconfig_roundtrips_through_storage():
    resolved, _ = resolve({"decoding": {"beam_size": 3}})
    assert RunConfig.model_validate(resolved).model_dump(mode="json") == resolved   # stored form re-validates


# ---------------- through the API ----------------
def _post(client, key, config=None, **extra):
    data = {k: v for k, v in {"config": json.dumps(config) if config is not None else None, **extra}.items() if v is not None}
    return client.post("/audio", headers={"X-API-Key": key}, data=data, files={"file": ("a.wav", make_wav(0.2), "audio/wav")})


def _wait_done(client, key, run_id):
    end = time.time() + 5
    while time.time() < end:
        r = client.get(f"/runs/{run_id}", headers={"X-API-Key": key}).json()
        if r["status"] in ("done", "failed"):
            return r
        time.sleep(0.02)
    raise AssertionError("timeout")


def test_bad_config_is_422_with_details_and_stores_nothing(client, make_user):
    _, key = make_user("alice")
    r = _post(client, key, {"decoding": {"beem_size": 3}})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["decoding", "beem_size"]
    assert list(client.app.state.storage.list_keys()) == []       # rejected before anything was stored


def test_run_exposes_resolved_config_and_hash(client, make_user):
    _, key = make_user("alice")
    sub = _post(client, key, {"decoding": {"beam_size": 3}}).json()
    run = _wait_done(client, key, sub["run_id"])
    assert run["config"]["decoding"]["beam_size"] == 3 and run["config"]["chunking"]["vad"]["backend"] == "silero"
    assert len(run["config_hash"]) == 64
    same = _wait_done(client, key, _post(client, key, {"decoding": {"beam_size": 3}}).json()["run_id"])
    assert same["config_hash"] == run["config_hash"]


def test_leaderboard_separates_same_model_with_different_settings(client, make_user):
    _, key = make_user("alice")
    ids = []
    for beam in (5, 5, 1):
        ids.append(_post(client, key, {"decoding": {"beam_size": beam}},
                         dataset="d", reference="this is a dummy transcript").json()["run_id"])
    for rid in ids:
        _wait_done(client, key, rid)
    lb = client.get("/datasets/d/leaderboard", headers={"X-API-Key": key}).json()
    assert len(lb) == 2 and {r["runs"] for r in lb} == {2, 1}
    assert {r["config"]["decoding"]["beam_size"] for r in lb} == {5, 1}
    assert all(r["model_version"] == "stub" and len(r["config_hash"]) == 64 for r in lb)


def test_force_error_option_still_works_through_validated_config(client, make_user):
    _, key = make_user("alice")
    run = _wait_done(client, key, _post(client, key, {"model_options": {"force_error": True}}).json()["run_id"])
    assert run["status"] == "failed" and "forced stub failure" in run["error"]


def test_config_schema_endpoint(client, make_user):
    _, key = make_user("alice")
    body = client.get("/config/schema", headers={"X-API-Key": key}).json()
    assert body["defaults"]["decoding"]["beam_size"] == 5
    assert "force_error" in body["models"]["stub"]["properties"]
    assert "properties" in body["json_schema"]
    assert client.get("/config/schema").status_code == 401


# ---------------- legacy rows (created before config validation existed) ----------------
def _seed_run(client, config, config_hash=None):
    from app.db.repositories import AudioRepo, RunRepo, UserRepo
    import io
    st = client.app.state
    with st.session_factory() as s:
        u, key = UserRepo(s).create(f"legacy{abs(hash(str(config)))}")
        st.storage.save_audio(f"audio/legacy/{u.id}.wav", io.BytesIO(make_wav(0.2)))
        a = AudioRepo(s).add(user_id=u.id, storage_key=f"audio/legacy/{u.id}.wav", checksum="c", duration=0.2,
                             original_sample_rate=44100, size_bytes=1)
        rid = RunRepo(s).add(a.id, "stub", config, config_hash=config_hash).id
        s.commit()
    return key, rid


def test_legacy_empty_config_run_still_processes(client):
    key, rid = _seed_run(client, {})
    client.app.state.queue.enqueue(rid)
    run = _wait_done(client, key, rid)
    assert run["status"] == "done" and run["config_hash"] is None


def test_legacy_unknown_option_fails_cleanly_instead_of_crashing_the_worker(client):
    key, rid = _seed_run(client, {"force_error": True})      # old-style key, now invalid
    client.app.state.queue.enqueue(rid)
    run = _wait_done(client, key, rid)
    assert run["status"] == "failed" and "force_error" in run["error"]
