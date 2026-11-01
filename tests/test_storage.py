import io
import hashlib
import tracemalloc

import pytest

from app.storage.base import BlobExistsError, BlobNotFoundError, InvalidKeyError
from app.storage.local import LocalDiskStorage
from tests.s3_helpers import new_s3_storage, wipe

# Every test below runs against every backend: if S3 passes the same suite as local disk, the two are interchangeable.
@pytest.fixture(params=["local", "s3"])
def storage(request, tmp_path):
    if request.param == "local":
        yield LocalDiskStorage(tmp_path / "blobs")
    else:
        st = new_s3_storage()          # skipped (with a reason) when no S3/MinIO server is reachable
        yield st
        wipe(st)


def test_roundtrip_and_checksum(storage):
    data = b"hello audio" * 1000
    info = storage.save_audio("audio/2026-10-23/a.wav", io.BytesIO(data))
    assert info.size_bytes == len(data)
    assert info.sha256 == hashlib.sha256(data).hexdigest()
    with storage.get_audio("audio/2026-10-23/a.wav") as f:
        assert f.read() == data


def test_exists_list_delete(storage):
    storage.save_audio("audio/d1/a.wav", io.BytesIO(b"1"))
    storage.save_audio("audio/d2/b.wav", io.BytesIO(b"2"))
    assert storage.exists("audio/d1/a.wav")
    assert list(storage.list_keys("audio/d1")) == ["audio/d1/a.wav"]
    assert sorted(storage.list_keys("audio/")) == ["audio/d1/a.wav", "audio/d2/b.wav"]
    assert storage.delete("audio/d1/a.wav") is True
    assert storage.delete("audio/d1/a.wav") is False
    assert not storage.exists("audio/d1/a.wav")


def test_immutable(storage):
    storage.save_audio("audio/x.wav", io.BytesIO(b"one"))
    with pytest.raises(BlobExistsError):
        storage.save_audio("audio/x.wav", io.BytesIO(b"two"))
    with storage.get_audio("audio/x.wav") as f:
        assert f.read() == b"one"


def test_missing(storage):
    with pytest.raises(BlobNotFoundError):
        storage.get_audio("audio/nope.wav")


@pytest.mark.parametrize("bad", ["../evil", "/abs/path", "audio/../../evil", ""])
def test_rejects_bad_keys(storage, bad):
    with pytest.raises(InvalidKeyError):
        storage.save_audio(bad, io.BytesIO(b"x"))


def test_failed_upload_leaves_no_partial_blob(storage):
    class Boom:
        def __init__(self): self.n = 0
        def read(self, _):
            self.n += 1
            if self.n > 2:
                raise IOError("connection dropped")
            return b"x" * 1024
    with pytest.raises(IOError):
        storage.save_audio("audio/partial.wav", Boom())
    assert not storage.exists("audio/partial.wav")
    assert list(storage.list_keys()) == []


def test_large_file_streams_with_flat_memory(storage):
    class Zeros:
        def __init__(self, total): self.left = total
        def read(self, n):
            n = min(n, self.left); self.left -= n
            return b"\0" * n
    total = 200 * 1024 * 1024
    tracemalloc.start()
    info = storage.save_audio("audio/big.wav", Zeros(total))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert info.size_bytes == total
    assert peak < 20 * 1024 * 1024  # nowhere near 200MB
