import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from app.db.models import User
from app.services import audio_service as svc

from .deps import get_current_user, get_service
from app.evaluation.service import leaderboard

from .deps import get_db
from .schemas import (LeaderboardRow, ReferenceBody, ReferenceResponse, ResultResponse,
                      RunStatusResponse, SubmitResponse)

router = APIRouter()


@router.get("/healthz")
def healthz():
    return {"ok": True}


@router.post("/audio", status_code=202, response_model=SubmitResponse)
def submit_audio(
    request: Request,
    file: UploadFile = File(...),
    model_version: str = Form("stub"),
    config: str = Form("{}"),
    dataset: str | None = Form(None),
    reference: str | None = Form(None),
    user: User = Depends(get_current_user),
    service: svc.AudioService = Depends(get_service),
):
    try:
        cfg = json.loads(config)
        if not isinstance(cfg, dict):
            raise ValueError
    except ValueError:
        raise HTTPException(422, "config must be a JSON object")
    try:
        audio, run = service.submit_audio(user, file.filename or "", file.file, model_version, cfg,
                                          dataset_name=dataset, reference_text=reference)
    except svc.UnsupportedFormat as e:
        raise HTTPException(415, str(e))
    except svc.UploadTooLarge as e:
        raise HTTPException(413, str(e))
    except svc.InvalidAudio as e:
        raise HTTPException(422, f"Could not decode audio: {e}")
    except svc.UnknownModel as e:
        raise HTTPException(422, f"Unknown model_version: {e}")
    request.app.state.queue.enqueue(run.id)  # after commit so the worker can see the row
    return SubmitResponse(audio_id=audio.id, run_id=run.id, status=run.status.value)


def _status_payload(run) -> RunStatusResponse:
    return RunStatusResponse(
        id=run.id, audio_id=run.audio_id, model_version=run.model_version,
        status=run.status.value, error=run.error, latency=run.latency, created_at=run.created_at,
        wer=run.wer, cer=run.cer,
    )


@router.get("/runs/{run_id}", response_model=RunStatusResponse)
def get_run(run_id: str, user: User = Depends(get_current_user), service: svc.AudioService = Depends(get_service)):
    try:
        return _status_payload(service.get_status(user, run_id))
    except svc.NotFound:
        raise HTTPException(404, "Run not found")


@router.get("/runs/{run_id}/result", response_model=ResultResponse)
def get_result(run_id: str, user: User = Depends(get_current_user), service: svc.AudioService = Depends(get_service)):
    try:
        run = service.get_result(user, run_id)
    except svc.NotFound:
        raise HTTPException(404, "Run not found")
    except svc.NotReady as e:
        raise HTTPException(409, f"Run is {e}, not done")
    return ResultResponse(run_id=run.id, model_version=run.model_version,
                          transcript=run.transcript or "", latency=run.latency,
                          wer=run.wer, cer=run.cer)


@router.put("/audio/{audio_id}/reference", response_model=ReferenceResponse)
def put_reference(audio_id: str, body: ReferenceBody, user: User = Depends(get_current_user),
                  service: svc.AudioService = Depends(get_service)):
    try:
        n = service.set_reference(user, audio_id, body.text, body.language, body.source)
    except svc.NotFound:
        raise HTTPException(404, "Audio not found")
    return ReferenceResponse(audio_id=audio_id, runs_rescored=n)


@router.get("/datasets/{name}/leaderboard", response_model=list[LeaderboardRow])
def dataset_leaderboard(name: str, user: User = Depends(get_current_user), db=Depends(get_db)):
    rows = leaderboard(db, user.id, name)
    if rows is None:
        raise HTTPException(404, "Dataset not found")
    return rows
