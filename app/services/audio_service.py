import os
from typing import BinaryIO

import soundfile as sf
from sqlalchemy.orm import Session

from app.config import Settings
from sqlalchemy import select

from app.db.models import Audio, Dataset, ReferenceTranscript, TranscriptionRun, User
from app.evaluation.service import evaluate_audio_runs
from app.db.repositories import AudioRepo, RunRepo
from app.pipeline.registry import ModelRegistry
from app.pipeline.run_config import ConfigInvalid, resolve_run_config
from app.storage.base import StorageBackend
from app.storage.keys import new_audio_key

ALLOWED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}


class ServiceError(Exception): ...
class UnsupportedFormat(ServiceError): ...
class InvalidAudio(ServiceError): ...
class UploadTooLarge(ServiceError): ...
class UnknownModel(ServiceError): ...


class InvalidConfig(ServiceError):
    def __init__(self, errors: list[dict]):
        super().__init__(errors)
        self.errors = errors
class NotFound(ServiceError): ...
class NotReady(ServiceError): ...


class _LimitedStream:
    """Enforces a max size while streaming, so we never buffer the upload."""

    def __init__(self, stream: BinaryIO, limit: int):
        self.stream, self.limit, self.read_bytes = stream, limit, 0

    def read(self, n: int = -1) -> bytes:
        data = self.stream.read(n)
        self.read_bytes += len(data)
        if self.read_bytes > self.limit:
            raise UploadTooLarge(f"upload exceeds {self.limit} bytes")
        return data


class AudioService:
    def __init__(self, session: Session, storage: StorageBackend, registry: ModelRegistry, settings: Settings):
        self.s, self.storage, self.registry, self.settings = session, storage, registry, settings
        self.audio_repo, self.run_repo = AudioRepo(session), RunRepo(session)

    def submit_audio(self, user: User, filename: str, stream: BinaryIO,
                     model_version: str, config: dict, dataset_name: str | None = None,
                     reference_text: str | None = None) -> tuple[Audio, TranscriptionRun]:
        ext = os.path.splitext(filename or "")[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise UnsupportedFormat(f"extension {ext!r} not in {sorted(ALLOWED_EXTENSIONS)}")
        if model_version not in self.registry.names():
            raise UnknownModel(model_version)
        try:  # validate before anything is stored
            resolved_config, cfg_hash = resolve_run_config(
                model_version, config, self.registry.options_model(model_version))
        except ConfigInvalid as e:
            raise InvalidConfig(e.errors) from e

        key = new_audio_key(ext)
        blob = self.storage.save_audio(key, _LimitedStream(stream, self.settings.max_upload_bytes))
        try:
            with self.storage.get_audio(key) as f:
                info = sf.info(f)
        except Exception as e:  # not decodable audio
            self.storage.delete(key)
            raise InvalidAudio(str(e)) from e

        dataset_id = self._get_or_create_dataset(user, dataset_name).id if dataset_name else None
        audio = self.audio_repo.add(
            user_id=user.id, storage_key=key, checksum=blob.sha256, duration=info.duration,
            original_sample_rate=info.samplerate, size_bytes=blob.size_bytes, dataset_id=dataset_id,
        )
        if reference_text is not None:
            self.s.add(ReferenceTranscript(audio_id=audio.id, text=reference_text))
        run = self.run_repo.add(audio.id, model_version, resolved_config, config_hash=cfg_hash)
        self.s.commit()
        return audio, run

    def _get_or_create_dataset(self, user: User, name: str) -> Dataset:
        ds = self.s.scalar(select(Dataset).where(Dataset.user_id == user.id, Dataset.name == name))
        if ds is None:
            ds = Dataset(user_id=user.id, name=name)
            self.s.add(ds)
            self.s.flush()
        return ds

    def set_reference(self, user: User, audio_id: str, text: str,
                      language: str | None = None, source: str | None = None) -> int:
        """Create or replace the reference for an audio file and re-score its finished runs.
        Returns how many runs were (re)scored."""
        audio = self.s.scalar(select(Audio).where(Audio.id == audio_id, Audio.user_id == user.id))
        if audio is None:
            raise NotFound(audio_id)
        ref = self.s.scalar(select(ReferenceTranscript).where(ReferenceTranscript.audio_id == audio_id))
        if ref is None:
            self.s.add(ReferenceTranscript(audio_id=audio_id, text=text, language=language, source=source))
        else:
            ref.text, ref.language, ref.source = text, language, source
        self.s.flush()
        n = evaluate_audio_runs(self.s, audio_id, self.settings.normalizer)
        self.s.commit()
        return n

    def get_status(self, user: User, run_id: str) -> TranscriptionRun:
        run = self.run_repo.get_for_user(run_id, user.id)
        # End the read transaction now so the connection goes back to the pool immediately,
        # instead of waiting for request teardown (which needs a free threadpool slot).
        self.s.commit()
        if run is None:
            raise NotFound(run_id)
        return run

    def get_result(self, user: User, run_id: str) -> TranscriptionRun:
        run = self.get_status(user, run_id)
        if run.status.value != "done":
            raise NotReady(run.status.value)
        return run
