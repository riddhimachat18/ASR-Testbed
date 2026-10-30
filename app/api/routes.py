import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from app.db.models import User
from app.services import audio_service as svc

from .deps import get_current_user, get_service
from .schemas import ResultResponse, RunStatusResponse, SubmitResponse

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
        audio, run = service.submit_audio(user, file.filename or "", file.file, model_version, cfg)
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
                          transcript=run.transcript or "", latency=run.latency)
