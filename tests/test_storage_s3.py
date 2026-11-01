import hashlib
import io
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import soundfile as sf
from botocore.exceptions import ClientError

from app.storage.base import BlobExistsError, InvalidKeyError, validate_key
from app.storage.local import LocalDiskStorage
from app.storage.s3 import MIN_PART, S3Storage
from app.storage.transfer import copy_blobs
from tests.s3_helpers import new_s3_storage, require_s3, wipe

MB = 1024 * 1024


@pytest.fixture
def s3():
    cfg = require_s3()
    st = S3Storage(**cfg, prefix=f"pytest-{uuid.uuid4().hex}/")
    st.ensure_bucket()
    yield st
    wipe(st)


def rand(n, seed=0) -> bytes:
    return np.random.default_rng(seed).integers(0, 256, n, dtype="uint8").tobytes()


# ------------------------------------------------------------------ key contract (no server needed)
@pytest.mark.parametrize("bad", ["", "/abs", "../x", "a/../b", "a/./b", "a//b", "a/", "a\\b", "a\x00b", "k" * 1025])
def test_validate_key_rejects(bad):
    with pytest.raises(InvalidKeyError):
        validate_key(bad)


@pytest.mark.parametrize("good", ["a", "audio/2026-10-31/abc.wav", "derived/ab/" + "0" * 64 + "/resampled-1.wav", "ünïcode/ok"])
def test_validate_key_accepts(good):
    assert validate_key(good) == good


def test_s3_backend_validates_keys_before_any_network_call():
    st = S3Storage("bucket-that-never-gets-contacted", endpoint_url="http://127.0.0.1:1")
    for bad in ("../evil", "/abs", ""):
        with pytest.raises(InvalidKeyError):
            st.save_audio(bad, io.BytesIO(b"x"))
        with pytest.raises(InvalidKeyError):
            st.exists(bad)


def test_part_size_schedule_and_limits():
    st = S3Storage("b", endpoint_url="http://127.0.0.1:1")
    assert st._part_size(1) == st._part_size(1000) == MIN_PART
    assert st._part_size(1001) == 2 * MIN_PART and st._part_size(10_000) == 10 * MIN_PART
    capacity = sum(st._part_size(n) for n in range(1, 10_001))
    assert capacity > 200 * 1024 ** 3                      # 10,000 parts reach well past 200 GB
    with pytest.raises(ValueError):
        S3Storage("b", part_size=MIN_PART - 1)


# ------------------------------------------------------------------ uploads
@pytest.mark.parametrize("size", [0, 1, MIN_PART - 1, MIN_PART, MIN_PART + 1, 2 * MIN_PART, 2 * MIN_PART + 7])
def test_sizes_around_part_boundaries_roundtrip(s3, size):
    data = rand(size, seed=size)
    info = s3.save_audio("audio/x.bin", io.BytesIO(data))
    assert info.size_bytes == size and info.sha256 == hashlib.sha256(data).hexdigest()
    with s3.get_audio("audio/x.bin") as f:
        assert hashlib.sha256(f.read()).hexdigest() == info.sha256


def test_short_reads_from_the_source_stream_are_handled(s3):
    data = rand(MIN_PART + 12345)

    class Trickle:                                          # like a network stream: never returns more than 100 KB
        def __init__(self): self.b = io.BytesIO(data)
        def read(self, n): return self.b.read(min(n, 100_000))

    assert s3.save_audio("audio/t.bin", Trickle()).sha256 == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("size", [100, 2 * MIN_PART + 1])
def test_concurrent_writers_to_one_key_exactly_one_wins(s3, size):
    if size > MIN_PART and os.environ.get("S3_TEST_ATOMIC_MULTIPART") == "0":
        pytest.skip("server does not make concurrent multipart completion atomic (known for some mock servers)")
    payloads = [rand(size, seed=i) for i in range(6)]
    barrier, results = threading.Barrier(6), []

    def write(i):
        barrier.wait()
        try:
            s3.save_audio("audio/race.bin", io.BytesIO(payloads[i])); return ("won", i)
        except BlobExistsError:
            return ("lost", i)

    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(write, range(6)))
    winners = [i for r, i in results if r == "won"]
    assert len(winners) == 1                               # the conditional write makes the race safe
    with s3.get_audio("audio/race.bin") as f:
        assert f.read() == payloads[winners[0]]            # and the stored content is exactly the winner's


def test_failed_multipart_upload_is_aborted_and_leaves_nothing(s3):
    class Dies:
        def __init__(self): self.left = 2 * MIN_PART + 10
        def read(self, n):
            if self.left <= 0:
                raise IOError("connection dropped")
            n = min(n, self.left); self.left -= n
            return b"x" * n

    with pytest.raises(IOError):
        s3.save_audio("audio/dies.bin", Dies())
    assert not s3.exists("audio/dies.bin") and list(s3.list_keys()) == []
    uploads = s3._client.list_multipart_uploads(Bucket=s3.bucket, Prefix=s3.prefix).get("Uploads", [])
    assert uploads == []                                   # no orphaned parts left to pay for


def test_server_errors_are_not_mistaken_for_existing_blobs():
    cfg = dict(require_s3(), bucket=f"asr-missing-{uuid.uuid4().hex[:10]}")
    st = S3Storage(**cfg, prefix="p/")
    with pytest.raises(ClientError):
        st.save_audio("a.bin", io.BytesIO(b"x"))           # NoSuchBucket must surface, not become BlobExistsError


# ------------------------------------------------------------------ seekable, range-based reads
def test_stream_is_seekable_and_random_access_is_correct(s3):
    data = rand(12 * MB)
    s3.save_audio("audio/big.bin", io.BytesIO(data))
    with s3.get_audio("audio/big.bin") as f:
        assert f.seekable() and f.readable() and f.tell() == 0
        assert f.seek(0, io.SEEK_END) == len(data) and f.read(10) == b""
        for off in (0, 1, 4 * MB - 5, 4 * MB, 7 * MB + 123, len(data) - 10):
            f.seek(off)
            assert f.read(100) == data[off:off + 100] and f.tell() == min(off + 100, len(data))
        f.seek(-10, io.SEEK_END)
        assert f.read() == data[-10:]
        f.seek(5); f.seek(7, io.SEEK_CUR)
        assert f.read(3) == data[12:15]
        with pytest.raises(OSError):
            f.seek(-1)
        f.seek(0)
        assert f.read() == data


def test_reading_a_slice_does_not_download_the_whole_object(s3):
    s3.save_audio("audio/big.bin", io.BytesIO(rand(12 * MB)))
    fetched = []
    s3._client.meta.events.register("after-call.s3.GetObject", lambda parsed, **kw: fetched.append(parsed["ContentLength"]))
    with s3.get_audio("audio/big.bin") as f:
        f.read(100)
        assert len(fetched) == 1 and fetched[0] <= s3.read_buffer        # one buffer, not 12 MB
        f.seek(-10, io.SEEK_END)
        f.read(10)
    assert sum(fetched) < 5 * MB


def test_soundfile_reads_a_wav_straight_from_s3(s3):
    t = np.arange(44100 * 3) / 44100
    samples = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype("int16")
    buf = io.BytesIO(); sf.write(buf, samples, 44100, format="WAV", subtype="PCM_16")
    s3.save_audio("audio/tone.wav", io.BytesIO(buf.getvalue()))
    with s3.get_audio("audio/tone.wav") as f:
        info = sf.info(f)
        assert (info.samplerate, info.channels, round(info.duration, 2)) == (44100, 1, 3.0)
        f.seek(0)
        out, _ = sf.read(f, dtype="int16")
    assert np.array_equal(out, samples)


def test_prefix_isolates_environments_and_is_invisible_to_callers(s3):
    other = S3Storage(**require_s3(), prefix=f"pytest-{uuid.uuid4().hex}/")
    try:
        s3.save_audio("audio/a.wav", io.BytesIO(b"1"))
        other.save_audio("audio/b.wav", io.BytesIO(b"2"))
        assert list(s3.list_keys()) == ["audio/a.wav"] and list(other.list_keys()) == ["audio/b.wav"]
        assert not s3.exists("audio/b.wav") and not other.exists("audio/a.wav")
    finally:
        wipe(other)


def test_listing_paginates_past_one_page(s3):
    for i in range(1005):                                  # S3 returns at most 1000 keys per page
        s3.save_audio(f"audio/many/{i:04d}.bin", io.BytesIO(b"x"))
    keys = list(s3.list_keys("audio/many/"))
    assert len(keys) == 1005 and keys == sorted(keys)


# ------------------------------------------------------------------ migration between backends
def _populate(storage):
    for key, size in {"audio/2026-10-30/a.wav": 1000, "audio/2026-10-31/b.wav": MIN_PART + 5, "derived/ab/c.json": 10}.items():
        storage.save_audio(key, io.BytesIO(rand(size, seed=size)))


def test_copy_blobs_local_to_local_is_idempotent_and_detects_mismatches(tmp_path):
    src, dst = LocalDiskStorage(tmp_path / "src"), LocalDiskStorage(tmp_path / "dst")
    _populate(src)
    dry = copy_blobs(src, dst, dry_run=True)
    assert len(dry.planned) == 3 and dst.exists("audio/2026-10-30/a.wav") is False        # nothing written
    first = copy_blobs(src, dst)
    assert first.copied == 3 and first.bytes_copied == 1000 + MIN_PART + 5 + 10 and not first.mismatched
    again = copy_blobs(src, dst)
    assert again.copied == 0 and again.skipped_existing == 3
    dst.delete("derived/ab/c.json"); dst.save_audio("derived/ab/c.json", io.BytesIO(b"tampered"))
    assert copy_blobs(src, dst).mismatched == ["derived/ab/c.json"]                       # verification catches it
    assert len(copy_blobs(src, dst, prefix="audio/").planned) == 0                       # prefix filter respected


def test_copy_blobs_local_to_s3(tmp_path, s3):
    src = LocalDiskStorage(tmp_path / "src")
    _populate(src)
    r = copy_blobs(src, s3)
    assert r.copied == 3 and not r.mismatched
    assert sorted(s3.list_keys()) == sorted(src.list_keys())
    with src.get_audio("audio/2026-10-31/b.wav") as a, s3.get_audio("audio/2026-10-31/b.wav") as b:
        assert a.read() == b.read()
    assert copy_blobs(src, s3).skipped_existing == 3


# ------------------------------------------------------------------ the whole app on S3
def test_whole_pipeline_runs_on_s3_storage(settings, make_user):
    import json, time
    from fastapi.testclient import TestClient
    from app.config import Settings
    from app.db.models import Base
    from app.db.repositories import UserRepo
    from app.main import create_app
    cfg = require_s3()
    prefix = f"pytest-{uuid.uuid4().hex}/"
    app = create_app(Settings(**{**settings.model_dump(), "storage_backend": "s3", "s3_bucket": cfg["bucket"],
                                 "s3_endpoint_url": cfg["endpoint_url"], "s3_access_key_id": cfg["access_key_id"],
                                 "s3_secret_access_key": cfg["secret_access_key"], "s3_prefix": prefix}))
    Base.metadata.create_all(app.state.engine)
    try:
        with TestClient(app) as c:
            with app.state.session_factory() as s:
                _, key = UserRepo(s).create("alice"); s.commit()
            buf = io.BytesIO(); sf.write(buf, np.zeros(8000 * 70, dtype="int16"), 8000, format="WAV", subtype="PCM_16")
            fixed = json.dumps({"chunking": {"strategy": "fixed", "max_chunk_s": 30, "overlap_s": 2}})

            def run(**data):
                r = c.post("/audio", headers={"X-API-Key": key}, data=data, files={"file": ("a.wav", buf.getvalue(), "audio/wav")})
                assert r.status_code == 202, r.text
                end = time.time() + 15
                while time.time() < end:
                    st = c.get(f"/runs/{r.json()['run_id']}", headers={"X-API-Key": key}).json()
                    if st["status"] in ("done", "failed"):
                        return st
                    time.sleep(0.05)
                raise AssertionError("timeout")

            assert run()["status"] == "done"                                             # original stored in S3, probed, read by worker
            assert run(model_version="stub-prepared", config=fixed)["status"] == "done"  # resample + chunks cached in S3
            storage = app.state.storage
            assert isinstance(storage, S3Storage)
            assert len(list(storage.list_keys("audio/"))) == 2 and len(list(storage.list_keys("derived/"))) == 2
    finally:
        wipe(app.state.storage)
