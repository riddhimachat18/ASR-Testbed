import hashlib
import secrets

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .models import Audio, RunStatus, TranscriptionRun, User

ALLOWED_TRANSITIONS: dict[RunStatus, set[RunStatus]] = {
    RunStatus.queued: {RunStatus.processing, RunStatus.failed},
    RunStatus.processing: {RunStatus.done, RunStatus.failed},
    RunStatus.done: set(),
    RunStatus.failed: {RunStatus.queued},  # manual retry
}


class InvalidTransition(Exception): ...


def hash_api_key(key: str) -> str:
    # API keys are long random tokens, so a fast hash is appropriate (no password-style KDF needed).
    return hashlib.sha256(key.encode()).hexdigest()


class UserRepo:
    def __init__(self, s: Session):
        self.s = s

    def create(self, username: str) -> tuple[User, str]:
        """Returns (user, plaintext_key). The plaintext is shown once and never stored."""
        key = secrets.token_urlsafe(32)
        user = User(username=username, hashed_api_key=hash_api_key(key))
        self.s.add(user)
        self.s.flush()
        return user, key

    def get_by_api_key(self, key: str) -> User | None:
        return self.s.scalar(select(User).where(User.hashed_api_key == hash_api_key(key)))


class AudioRepo:
    def __init__(self, s: Session):
        self.s = s

    def add(self, **fields) -> Audio:
        audio = Audio(**fields)
        self.s.add(audio)
        self.s.flush()
        return audio

    def get(self, audio_id: str) -> Audio | None:
        return self.s.get(Audio, audio_id)


class RunRepo:
    def __init__(self, s: Session):
        self.s = s

    def add(self, audio_id: str, model_version: str, config: dict) -> TranscriptionRun:
        run = TranscriptionRun(audio_id=audio_id, model_version=model_version, config=config)
        self.s.add(run)
        self.s.flush()
        return run

    def get(self, run_id: str) -> TranscriptionRun | None:
        return self.s.get(TranscriptionRun, run_id)

    def get_for_user(self, run_id: str, user_id: str) -> TranscriptionRun | None:
        return self.s.scalar(
            select(TranscriptionRun)
            .join(Audio, Audio.id == TranscriptionRun.audio_id)
            .where(TranscriptionRun.id == run_id, Audio.user_id == user_id)
        )

    def transition(self, run: TranscriptionRun, new: RunStatus) -> None:
        if new not in ALLOWED_TRANSITIONS[run.status]:
            raise InvalidTransition(f"{run.status.value} -> {new.value}")
        run.status = new

    def claim(self, run_id: str) -> bool:
        """Atomic queued->processing. Only one worker can win; duplicates/retries are harmless."""
        res = self.s.execute(
            update(TranscriptionRun)
            .where(TranscriptionRun.id == run_id, TranscriptionRun.status == RunStatus.queued)
            .values(status=RunStatus.processing)
        )
        return res.rowcount == 1
    
    def complete(self, run_id: str, transcript: str, latency: float) -> bool:
        """Atomic processing->done. Returns False (and writes nothing) if the run is no longer
        'processing', e.g. another worker already finished it. First writer wins."""
        res = self.s.execute(
            update(TranscriptionRun)
            .where(TranscriptionRun.id == run_id, TranscriptionRun.status == RunStatus.processing)
            .values(status=RunStatus.done, transcript=transcript, latency=latency)
        )
        return res.rowcount == 1

    def mark_failed(self, run_id: str, error: str, latency: float) -> bool:
        """Atomic processing->failed, same guard as complete(). Never raises on a state mismatch."""
        res = self.s.execute(
            update(TranscriptionRun)
            .where(TranscriptionRun.id == run_id, TranscriptionRun.status == RunStatus.processing)
            .values(status=RunStatus.failed, error=error, latency=latency)
        )
        return res.rowcount == 1

    def finish(self, run: TranscriptionRun, transcript: str, latency: float) -> None:
        self.transition(run, RunStatus.done)
        run.transcript, run.latency = transcript, latency

    def fail(self, run: TranscriptionRun, error: str, latency: float) -> None:
        self.transition(run, RunStatus.failed)
        run.error, run.latency = error, latency
