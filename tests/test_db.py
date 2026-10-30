import pytest
from sqlalchemy.orm import sessionmaker

from app.db.models import Audio, Base, RunStatus
from app.db.repositories import AudioRepo, InvalidTransition, RunRepo, UserRepo
from app.db.session import make_engine


@pytest.fixture
def session(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path/'db.sqlite'}")
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as s:
        yield s


def _run(session):
    user, key = UserRepo(session).create("alice")
    audio = AudioRepo(session).add(user_id=user.id, storage_key="audio/a.wav", checksum="c",
                                   duration=1.0, original_sample_rate=44100, size_bytes=10)
    return user, key, RunRepo(session).add(audio.id, "stub", {})


def test_api_key_is_hashed_and_resolvable(session):
    user, key, _ = _run(session)
    assert user.hashed_api_key != key
    assert UserRepo(session).get_by_api_key(key).id == user.id
    assert UserRepo(session).get_by_api_key("wrong") is None


def test_valid_and_invalid_transitions(session):
    _, _, run = _run(session)
    repo = RunRepo(session)
    assert run.status is RunStatus.queued
    repo.transition(run, RunStatus.processing)
    repo.finish(run, "text", 0.1)
    assert run.status is RunStatus.done
    with pytest.raises(InvalidTransition):
        repo.transition(run, RunStatus.processing)


def test_claim_is_atomic_single_winner(session):
    _, _, run = _run(session)
    session.commit()
    repo = RunRepo(session)
    assert repo.claim(run.id) is True
    assert repo.claim(run.id) is False


def test_one_audio_many_runs(session):
    _, _, run = _run(session)
    RunRepo(session).add(run.audio_id, "other-model", {"beam": 5})
    session.commit()
    assert len(session.get(Audio, run.audio_id).runs) == 2

def test_complete_and_fail_are_guarded_first_writer_wins(session):
    _, _, run = _run(session)
    session.commit()
    repo = RunRepo(session)
    assert repo.complete(run.id, "x", 0.1) is False        # still queued: nothing written
    assert repo.claim(run.id) is True
    assert repo.complete(run.id, "first", 0.1) is True
    assert repo.complete(run.id, "second", 0.2) is False   # duplicate finisher is discarded
    assert repo.mark_failed(run.id, "late error", 0.3) is False  # failure path cannot overwrite done
    session.expire_all()
    assert repo.get(run.id).transcript == "first" and repo.get(run.id).status is RunStatus.done