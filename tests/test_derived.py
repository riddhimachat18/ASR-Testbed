import io
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import soundfile as sf
from sqlalchemy.orm import sessionmaker

from app.db.models import Base
from app.db.repositories import AudioRepo, UserRepo
from app.db.session import make_engine
from app.pipeline.derived import (DerivedCache, VadUnavailable, derived_key, plan_fixed_chunks,
                                  plan_vad_chunks, prune_orphans)
from app.pipeline.run_config import ChunkingConfig
from app.storage.local import LocalDiskStorage


class CountingStorage(LocalDiskStorage):
    def __init__(self, root):
        super().__init__(root)
        self.saves: list[str] = []

    def save_audio(self, key, stream):
        self.saves.append(key)
        return super().save_audio(key, stream)


@pytest.fixture
def storage(tmp_path):
    return CountingStorage(tmp_path / "blobs")


def wav_bytes(seconds=1.0, rate=44100, channels=1, freq=440.0, subtype="PCM_16", data=None) -> bytes:
    if data is None:
        t = np.arange(int(seconds * rate)) / rate
        sig = (0.5 * np.sin(2 * np.pi * freq * t)).astype("float32")
        data = np.stack([sig] * channels, axis=1) if channels > 1 else sig
    buf = io.BytesIO()
    sf.write(buf, data, rate, format="WAV", subtype=subtype)
    return buf.getvalue()


def put(storage, key, raw) -> str:
    return storage.save_audio(key, io.BytesIO(raw)).sha256


class FakeVad:
    name = "silero"

    def __init__(self, segments, version="1"):
        self.segments, self.version, self.seen_rate = segments, version, None

    def detect_speech_segments(self, audio, cfg):
        self.seen_rate = sf.info(audio).samplerate
        audio.seek(0)
        return self.segments


# ------------------------------------------------------------------ keys and planning
def test_key_layout_and_stability():
    cs = "ab" + "0" * 62
    k = derived_key(cs, "chunks", {"a": 1, "b": 2}, "json")
    assert k == derived_key(cs, "chunks", {"b": 2, "a": 1}, "json")        # key order cannot matter
    assert k.startswith(f"derived/ab/{cs}/chunks-") and k.endswith(".json")
    assert k != derived_key(cs, "chunks", {"a": 1, "b": 3}, "json")        # any parameter change = new key
    assert k != derived_key(cs, "resampled", {"a": 1, "b": 2}, "json")
    for bad in ("short", "A" * 64, "../" + "0" * 61, ""):
        with pytest.raises(ValueError):
            derived_key(bad, "chunks", {}, "json")


@pytest.mark.parametrize("duration,max_s,overlap,expected", [
    (70, 30, 2, [(0, 30), (28, 58), (56, 70)]),
    (30, 30, 0, [(0, 30)]),                    # exactly one window
    (12.5, 30, 0, [(0, 12.5)]),                # shorter than a window
    (60, 30, 0, [(0, 30), (30, 60)]),
    (0, 30, 0, []),
])
def test_plan_fixed_chunks(duration, max_s, overlap, expected):
    assert [(c.start, c.end) for c in plan_fixed_chunks(duration, max_s, overlap)] == expected


def test_plan_vad_chunks_force_cuts_long_segments():
    got = plan_vad_chunks([(10, 95), (1, 4)], 30)       # unsorted input is fine
    assert [(c.start, c.end) for c in got] == [(1, 4), (10, 40), (40, 70), (70, 95)]
    assert plan_vad_chunks([], 30) == []


# ------------------------------------------------------------------ resampled audio
def test_resample_stereo_44k_to_mono_16k_keeps_duration_and_pitch(storage):
    cs = put(storage, "audio/a.wav", wav_bytes(3.0, 44100, channels=2, freq=440))
    with DerivedCache(storage).resampled("audio/a.wav", cs, 16000) as f:
        info = sf.info(f)
        assert (info.samplerate, info.channels, info.subtype) == (16000, 1, "PCM_16")
        assert abs(info.duration - 3.0) < 0.01
        f.seek(0)
        y, _ = sf.read(f)
    peak = np.fft.rfftfreq(len(y), 1 / 16000)[np.argmax(np.abs(np.fft.rfft(y)))]
    assert abs(peak - 440) < 5


def test_resampled_is_built_once_then_reused(storage):
    cs = put(storage, "audio/a.wav", wav_bytes(1.0, 44100))
    cache = DerivedCache(storage)
    cache.resampled("audio/a.wav", cs, 16000).close()
    cache.resampled("audio/a.wav", cs, 16000).close()
    assert len([k for k in storage.saves if k.startswith("derived/")]) == 1
    cache.resampled("audio/a.wav", cs, 8000).close()                      # another rate = another artifact
    assert len([k for k in storage.saves if k.startswith("derived/")]) == 2


def test_same_rate_mono_is_a_lossless_passthrough(storage):
    samples = np.random.default_rng(0).integers(-20000, 20000, 16000, dtype="int16")
    cs = put(storage, "audio/a.wav", wav_bytes(data=samples, rate=16000, subtype="PCM_16"))
    with DerivedCache(storage).resampled("audio/a.wav", cs, 16000) as f:
        out, _ = sf.read(f, dtype="int16")
    assert np.array_equal(out, samples)


def test_identical_uploads_share_one_artifact(storage):
    raw = wav_bytes(1.0, 22050)
    cs1, cs2 = put(storage, "audio/one.wav", raw), put(storage, "audio/two.wav", raw)   # same bytes, two uploads
    assert cs1 == cs2
    cache = DerivedCache(storage)
    cache.resampled("audio/one.wav", cs1, 16000).close()
    cache.resampled("audio/two.wav", cs2, 16000).close()
    assert len([k for k in storage.saves if k.startswith("derived/")]) == 1


def test_concurrent_builds_leave_one_clean_artifact(storage, tmp_path):
    cs = put(storage, "audio/a.wav", wav_bytes(4.0, 44100))
    cache = DerivedCache(storage)
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda _: cache.resampled("audio/a.wav", cs, 16000).close(), range(6)))
    assert len(list(storage.list_keys("derived/"))) == 1
    assert not list((tmp_path / "blobs").rglob("*.tmp"))


# ------------------------------------------------------------------ chunk lists
FIXED = ChunkingConfig(strategy="fixed", max_chunk_s=30, overlap_s=2)


def derived_keys(storage):
    return sorted(storage.list_keys("derived/"))


def test_fixed_chunks_are_cached_and_parameter_sensitive(storage):
    cs = put(storage, "audio/a.wav", wav_bytes(70, 8000))
    cache = DerivedCache(storage)
    got = cache.chunks("audio/a.wav", cs, FIXED, 16000)
    assert [(c.start, c.end) for c in got] == [(0, 30), (28, 58), (56, 70)]
    assert len(derived_keys(storage)) == 2                                   # resampled + chunk list
    assert cache.chunks("audio/a.wav", cs, FIXED, 16000) == got
    assert len(derived_keys(storage)) == 2                                   # hit: nothing new
    cache.chunks("audio/a.wav", cs, ChunkingConfig(strategy="fixed", max_chunk_s=30, overlap_s=4), 16000)
    assert len(derived_keys(storage)) == 3                                   # new overlap: new chunk list only
    cache.chunks("audio/a.wav", cs, ChunkingConfig(strategy="fixed", max_chunk_s=20, overlap_s=2), 16000)
    assert len(derived_keys(storage)) == 4                                   # resampled audio is reused throughout


def test_chunk_entry_records_its_checksum_and_parameters(storage):
    cs = put(storage, "audio/a.wav", wav_bytes(40, 8000))
    DerivedCache(storage).chunks("audio/a.wav", cs, FIXED, 16000)
    key = next(k for k in derived_keys(storage) if "/chunks-" in k)
    with storage.get_audio(key) as f:
        data = json.load(f)
    assert data["source_sha256"] == cs and data["params"]["sample_rate"] == 16000
    assert data["params"]["chunking"]["strategy"] == "fixed" and data["duration_s"] == 40.0


@pytest.mark.parametrize("bad", [b"not json at all", json.dumps({"source_sha256": "0" * 64, "params": {}, "chunks": []}).encode()])
def test_corrupt_or_foreign_chunk_entry_is_rebuilt(storage, bad):
    cs = put(storage, "audio/a.wav", wav_bytes(40, 8000))
    cache = DerivedCache(storage)
    good = cache.chunks("audio/a.wav", cs, FIXED, 16000)
    key = next(k for k in derived_keys(storage) if "/chunks-" in k)
    storage.delete(key)
    storage.save_audio(key, io.BytesIO(bad))                                 # simulate damage
    assert cache.chunks("audio/a.wav", cs, FIXED, 16000) == good             # detected, rebuilt
    with storage.get_audio(key) as f:
        assert json.load(f)["source_sha256"] == cs


def test_vad_chunks_use_backend_and_its_version_is_part_of_the_key(storage):
    cs = put(storage, "audio/a.wav", wav_bytes(100, 8000))
    cache = DerivedCache(storage)
    vad = ChunkingConfig(strategy="vad", max_chunk_s=30)
    v1 = FakeVad([(1.0, 4.0), (10.0, 95.0)], version="1")
    got = cache.chunks("audio/a.wav", cs, vad, 16000, {"silero": v1})
    assert [(c.start, c.end) for c in got] == [(1, 4), (10, 40), (40, 70), (70, 95)]
    assert v1.seen_rate == 16000                                            # VAD sees the resampled audio
    n = len(derived_keys(storage))
    cache.chunks("audio/a.wav", cs, vad, 16000, {"silero": FakeVad([], version="1")})
    assert len(derived_keys(storage)) == n                                  # same version: cache hit (fake not consulted)
    cache.chunks("audio/a.wav", cs, vad, 16000, {"silero": FakeVad([(0.0, 5.0)], version="2")})
    assert len(derived_keys(storage)) == n + 1                              # upgraded library: new entry


def test_missing_vad_backend_raises_and_caches_nothing(storage):
    cs = put(storage, "audio/a.wav", wav_bytes(10, 8000))
    with pytest.raises(VadUnavailable, match="silero"):
        DerivedCache(storage).chunks("audio/a.wav", cs, ChunkingConfig(), 16000, {})
    assert derived_keys(storage) == []


# ------------------------------------------------------------------ pruning
def test_prune_orphans_removes_only_derived_data_of_deleted_audio(storage, tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'p.db'}")
    Base.metadata.create_all(engine)
    sf_ = sessionmaker(bind=engine, expire_on_commit=False)
    keep = put(storage, "audio/keep.wav", wav_bytes(1, 8000))
    gone = put(storage, "audio/gone.wav", wav_bytes(2, 8000))
    with sf_() as s:
        u = UserRepo(s).create("u")[0]
        AudioRepo(s).add(user_id=u.id, storage_key="audio/keep.wav", checksum=keep, duration=1,
                         original_sample_rate=8000, size_bytes=1)
        s.commit()                                                          # `gone` has no audio row any more
    cache = DerivedCache(storage)
    cache.resampled("audio/keep.wav", keep, 16000).close()
    cache.resampled("audio/gone.wav", gone, 16000).close()
    would = prune_orphans(storage, sf_, dry_run=True)
    assert len(would) == 1 and gone in would[0] and len(derived_keys(storage)) == 2     # dry run deletes nothing
    assert prune_orphans(storage, sf_) == would
    assert len(derived_keys(storage)) == 1 and keep in derived_keys(storage)[0]
    assert storage.exists("audio/gone.wav")                                  # originals are never touched


# ------------------------------------------------------------------ through the API and the worker
def _post(client, key, raw, **data):
    return client.post("/audio", headers={"X-API-Key": key}, data=data, files={"file": ("a.wav", raw, "audio/wav")})


def _wait_done(client, key, run_id):
    end = time.time() + 10
    while time.time() < end:
        r = client.get(f"/runs/{run_id}", headers={"X-API-Key": key}).json()
        if r["status"] in ("done", "failed"):
            return r
        time.sleep(0.02)
    raise AssertionError("timeout")


def test_worker_uses_the_cache_across_duplicate_uploads(client, make_user):
    _, key = make_user("alice")
    raw = wav_bytes(70, 8000)
    cfg = json.dumps({"chunking": {"strategy": "fixed", "max_chunk_s": 30, "overlap_s": 2}})
    storage = client.app.state.storage
    r1 = _wait_done(client, key, _post(client, key, raw, model_version="stub-prepared", config=cfg).json()["run_id"])
    assert r1["status"] == "done"
    res = client.get(f"/runs/{r1['id']}/result", headers={"X-API-Key": key}).json()
    assert res["transcript"] == "stub transcript from 3 chunks"
    assert len(list(storage.list_keys("derived/"))) == 2
    # the same file uploaded again is a new audio row and blob, but the same checksum: pure cache hits
    r2 = _wait_done(client, key, _post(client, key, raw, model_version="stub-prepared", config=cfg).json()["run_id"])
    assert r2["status"] == "done" and len(list(storage.list_keys("derived/"))) == 2
    cfg2 = json.dumps({"chunking": {"strategy": "fixed", "max_chunk_s": 30, "overlap_s": 4}})
    _wait_done(client, key, _post(client, key, raw, model_version="stub-prepared", config=cfg2).json()["run_id"])
    assert len(list(storage.list_keys("derived/"))) == 3                    # only the new chunk list was added


def test_plain_stub_model_does_not_touch_the_cache(client, make_user):
    _, key = make_user("alice")
    _wait_done(client, key, _post(client, key, wav_bytes(1, 8000)).json()["run_id"])
    assert list(client.app.state.storage.list_keys("derived/")) == []


def test_run_fails_cleanly_when_the_vad_backend_is_missing(client, make_user):
    _, key = make_user("alice")
    r = _wait_done(client, key, _post(client, key, wav_bytes(5, 8000), model_version="stub-prepared").json()["run_id"])
    assert r["status"] == "failed" and "VAD backend 'silero' is not available" in r["error"]
