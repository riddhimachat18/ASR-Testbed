import io
import math
import struct
import wave

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db.models import Base
from app.db.repositories import UserRepo
from app.main import create_app


def make_wav(seconds: float = 1.0, rate: int = 44100) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        n = int(seconds * rate)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(n)))
    return buf.getvalue()


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, queue_backend="inprocess", data_root=str(tmp_path / "blobs"), database_url=f"sqlite:///{tmp_path/'t.db'}",
                    stub_sleep_seconds=0.05, worker_threads=4, max_upload_bytes=5_000_000,
                    recovery_interval_s=0)


@pytest.fixture
def client(settings):
    app = create_app(settings)
    Base.metadata.create_all(app.state.engine)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def make_user(client):
    def _make(name):
        with client.app.state.session_factory() as s:
            user, key = UserRepo(s).create(name)
            s.commit()
            return user, key
    return _make
