# ASR Testbed (Phases 0-1 implemented, Phase 2 skeleton running)

## Run
    pip install -r requirements.txt
    cp .env.example .env
    alembic upgrade head
    python -m scripts.create_user alice          # prints an API key once
    uvicorn app.main:factory --factory --reload
    curl -H "X-API-Key: $KEY" -F file=@sample.wav -F model_version=stub localhost:8000/audio
    curl -H "X-API-Key: $KEY" localhost:8000/runs/<run_id>
    curl -H "X-API-Key: $KEY" localhost:8000/runs/<run_id>/result

## Test
    pytest

## Layout
    app/api        routes, auth dependency, schemas (never touches storage/DB directly)
    app/services   submit_audio / get_status / get_result
    app/storage    StorageBackend interface + LocalDiskStorage (S3/MinIO goes here next)
    app/db         SQLAlchemy models, repositories (status state machine, atomic claim)
    app/pipeline   ModelWrapper interface, stub model, registry, worker + queue
    migrations     Alembic

## Decisions made while implementing
- `list` on the storage interface is named `list_keys` (avoids shadowing the builtin).
- Duplicate uploads: every upload gets its own blob + audio row; `checksum` is indexed, not unique.
  Dedupe (reuse blob, new audio row) is a small change in `AudioService.submit_audio` once you want it.
- Accepted formats: .wav .flac .ogg .mp3, validated by decoding the header with soundfile.
- Workers claim runs with an atomic `UPDATE ... WHERE status='queued'`, so duplicate/retried jobs are no-ops.
- Queue is an in-process thread pool behind `enqueue(run_id)`; Celery/Redis replaces `InProcessQueue` only.
