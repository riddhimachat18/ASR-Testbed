# ASR Testbed (Phases 0-1 implemented, Phase 2 + Part E queue running)

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

## Part E: Celery + Redis and crash recovery
    docker compose up -d postgres redis
    alembic upgrade head                         # adds started_at / attempts / enqueued_at
    # .env: QUEUE_BACKEND=celery
    uvicorn app.main:factory --factory           # terminal 1 (API + recovery sweeper)
    celery -A app.celery_app worker --concurrency=4 --loglevel=info    # terminal 2 (add more terminals to scale)
    python -m scripts.recover                    # one manual sweep (--stuck-after 0 --queued-after 0 to force)

Reliability model (details in app/celery_app.py and app/pipeline/recovery.py):
- `acks_late` + `reject_on_worker_lost`: a message is acknowledged only when the task finishes.
- A crashed worker leaves its run in `processing`; the recovery sweep (API startup + every RECOVERY_INTERVAL_S) resets runs
  stuck longer than STUCK_AFTER_S back to `queued` and re-enqueues them. After MAX_ATTEMPTS the run is failed instead.
- `queued` runs not (re)enqueued for QUEUED_AFTER_S are re-published (covers a crash between DB commit and enqueue,
  or a broker that lost the message). Keep QUEUED_AFTER_S above your worst-case healthy queue wait, otherwise you
  publish harmless but wasteful duplicate messages.
- Duplicates are always safe: `claim()` and `complete()` are atomic, first writer wins.
- Invariants checked at startup: STUCK_AFTER_S and VISIBILITY_TIMEOUT_S must both exceed task soft + hard margin.
