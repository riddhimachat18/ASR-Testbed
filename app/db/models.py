import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (JSON, DateTime, Enum, Float, ForeignKey, Index,
                        Integer, String, Text, UniqueConstraint)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class RunStatus(str, enum.Enum):
    queued = "queued"
    processing = "processing"
    done = "done"
    failed = "failed"


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    username: Mapped[str] = mapped_column(String(100), unique=True)
    hashed_api_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Dataset(Base):
    """A named group of audio files with reference transcripts (e.g. LibriSpeech test-clean)."""
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_datasets_user_name"),)


class Audio(Base):
    __tablename__ = "audio"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    storage_key: Mapped[str] = mapped_column(String(300), unique=True)
    checksum: Mapped[str] = mapped_column(String(64), index=True)  # sha256; not unique (see README)
    duration: Mapped[float] = mapped_column(Float)
    original_sample_rate: Mapped[int] = mapped_column(Integer)
    size_bytes: Mapped[int] = mapped_column(Integer)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    dataset_id: Mapped[str | None] = mapped_column(
        ForeignKey("datasets.id", name="fk_audio_dataset_id"), nullable=True, index=True)
    # The file's ID in its source dataset (LibriSpeech utterance ID such as 1089-134686-0000). Audio ids differ
    # between databases, so exports and cross-machine comparisons match files on this (Part H).
    external_id: Mapped[str | None] = mapped_column(String(200), nullable=True)

    runs: Mapped[list["TranscriptionRun"]] = relationship(back_populates="audio")

    __table_args__ = (Index("ix_audio_dataset_external_id", "dataset_id", "external_id", unique=True),)


class ReferenceTranscript(Base):
    """Ground truth for one audio file. (Named reference_transcripts: `references` is an SQL keyword.)"""
    __tablename__ = "reference_transcripts"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    audio_id: Mapped[str] = mapped_column(ForeignKey("audio.id"), unique=True)
    text: Mapped[str] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    source: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class TranscriptionRun(Base):
    __tablename__ = "transcription_runs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    audio_id: Mapped[str] = mapped_column(ForeignKey("audio.id"), index=True)
    model_version: Mapped[str] = mapped_column(String(100))
    config: Mapped[dict] = mapped_column(JSON, default=dict)  # fully resolved RunConfig
    config_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)  # NULL = legacy run
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus, native_enum=False, length=20), default=RunStatus.queued, index=True
    )
    transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    # queue bookkeeping (Part E)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    enqueued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, default=_now)
    # evaluation (Part F): headline metrics as columns (cheap to sort/aggregate), details as JSON
    wer: Mapped[float | None] = mapped_column(Float, nullable=True)
    cer: Mapped[float | None] = mapped_column(Float, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # profiling (Part H): per-stage timings, cache hits and the runtime (device, library versions, GPU name).
    # Its own column because rescoring rewrites `metrics`, and timings must survive that.
    profile: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    audio: Mapped[Audio] = relationship(back_populates="runs")

    __table_args__ = (Index("ix_runs_status_created", "status", "created_at"),)
