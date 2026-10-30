import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (JSON, DateTime, Enum, Float, ForeignKey, Index,
                        Integer, String, Text)
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

    runs: Mapped[list["TranscriptionRun"]] = relationship(back_populates="audio")


class TranscriptionRun(Base):
    __tablename__ = "transcription_runs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    audio_id: Mapped[str] = mapped_column(ForeignKey("audio.id"), index=True)
    model_version: Mapped[str] = mapped_column(String(100))
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus, native_enum=False, length=20), default=RunStatus.queued, index=True
    )
    transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    audio: Mapped[Audio] = relationship(back_populates="runs")

    __table_args__ = (Index("ix_runs_status_created", "status", "created_at"),)
