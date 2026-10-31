from datetime import datetime

from pydantic import BaseModel


class SubmitResponse(BaseModel):
    audio_id: str
    run_id: str
    status: str


class RunStatusResponse(BaseModel):
    id: str
    audio_id: str
    model_version: str
    status: str
    error: str | None
    latency: float | None
    created_at: datetime
    wer: float | None = None
    cer: float | None = None
    config_hash: str | None = None
    config: dict | None = None


class ResultResponse(BaseModel):
    run_id: str
    model_version: str
    transcript: str
    latency: float | None
    wer: float | None = None
    cer: float | None = None


class ReferenceBody(BaseModel):
    text: str
    language: str | None = None
    source: str | None = None


class ReferenceResponse(BaseModel):
    audio_id: str
    runs_rescored: int


class LeaderboardRow(BaseModel):
    model_version: str
    config_hash: str | None
    config: dict | None
    runs: int
    audio_files: int
    ref_words: int
    wer: float
    cer: float
