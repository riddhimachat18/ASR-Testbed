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


class ResultResponse(BaseModel):
    run_id: str
    model_version: str
    transcript: str
    latency: float | None
