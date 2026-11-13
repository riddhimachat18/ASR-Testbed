# ASR Research Testbed

A pipeline for benchmarking speech recognition (WER, CER, latency, real-time factor) so that a difference in a
result comes from the method under test, not from the infrastructure. FastAPI, Celery + Redis, Postgres,
local disk or S3/MinIO, Silero VAD, faster-whisper, jiwer. One code path serves CPU and GPU machines; only
settings differ. Every design decision and result is recorded in the design log (Claude doc, "ASR Research
Testbed: Design Log").

## Setup (CPU, macOS)

    pip install torch torchaudio            # CPU wheels
    pip install -r requirements.txt
    cp .env.example .env                    # set DATABASE_URL to Postgres, QUEUE_BACKEND=celery
    docker compose up -d postgres redis
    alembic upgrade head
    python -m scripts.create_user alice     # or, on an existing database: python -m scripts.rotate_keys
    python scripts/import_librispeech.py --source LibriSpeech/test-clean
    uvicorn app.main:factory --factory
    FORKED_BY_MULTIPROCESSING=1 celery -A app.celery_app worker --concurrency=1 --loglevel=info

GPU machine: see [docs/GPU.md](docs/GPU.md) (`ASR_ENV_FILE=.env.gpu`, `DEVICE=cuda`, `COMPUTE_TYPE=float16`).

## Settings that decide what a run means

| Setting | Default | Effect |
|---|---|---|
| `DEVICE`, `COMPUTE_TYPE` | `cpu`, `int8` | Written into every faster-whisper run's `config.model_options`, so part of the config hash. Override per run with `model_options` |
| `model_options.temperature_fallback` | `false` | Whisper's random retry at higher temperatures. Off keeps decoding deterministic and compute bounded (decision in design log H.7) |
| `model_options.no_repeat_ngram_size`, `repetition_penalty` | `3`, `1.0` | Deterministic loop guard (per run, part of the hash; design log H.7) |
| `VAD_DEVICE` | `cpu` | Keep `cpu` on every machine: identical chunk boundaries |
| `NORMALIZER` | `whisper-en-v1` | Text normalization for new scores (`basic-v1` was used before Part H) |
| `ASR_ENV_FILE` | `.env` | Which env file to read |

## Workflow: measure, export, compare

    # 1. baseline one model + settings over the dataset (one worker for any speed number)
    python scripts/run_baseline.py --model faster-whisper-tiny --worker-concurrency 1
    python scripts/report_baseline.py results/CPU/baseline_*.json

    # 2. export it (per file, keyed by LibriSpeech utterance ID) and commit the export
    python -m scripts.export_runs --model faster-whisper-tiny --machine mac-cpu --worker-concurrency 1

    # 3. compare any two exports, from either machine: paired bootstrap, 95% interval
    python -m scripts.compare_runs results/exports/A.jsonl.gz results/exports/B.jsonl.gz \
        --markdown results/comparisons/A_vs_B.md

    # re-score finished runs under another normalization (export first: this overwrites the scores)
    python -m scripts.evaluate --recompute --normalization whisper-en-v1 --model faster-whisper-tiny

Rules: compare WER only under one normalization; compare speed only at equal worker concurrency; claim a
difference only when the paired-bootstrap interval excludes zero.

## Dashboard and tracing

Open **http://localhost:8000/dashboard** while the API is running. It shows, side by side for up to three run
groups (from this database or from committed exports such as the RTX 3050 runs):

- **Live pipeline:** queue counts, runs per minute, seconds of work per stage every 10 s, a timeline of runs per
  worker process, and the latest finished runs. Updated every second (server-sent events).
- **Where the time goes:** self time per stage (prepare, chunk slicing, audio decode, log-mel features,
  encoder, language detection, decoder, scoring, database) and per-call p50 / p95.
- **Linked explorer:** latency vs chunks, latency vs duration, WER vs SNR, encoder vs decoder time. Select runs
  in one chart and they light up in all of them; click a run for its span timeline and transcript diff.
- **Data and model internals:** distributions of duration, chunks, padding, speech ratio, SNR and WER; WER
  against decoder confidence and compression ratio; median cost per call.

Where the numbers come from:

- Every run attempt is traced (`app/observability/tracing.py`): each step is a timed span, nested, with what it
  processed (cache hit, frames, tokens, log-prob). Spans are stored in `stage_events` when the run finishes.
- faster-whisper's internals are timed by wrapping its own methods at model load (`instrument_model` in
  `app/pipeline/faster_whisper_wrapper.py`): audio decode, log-mel features, each encoder call, language
  detection, each decoder call. Inputs and outputs pass through unchanged.
- Audio features (`audio_features` table): level, clipping, speech ratio, estimated SNR, spectral centroid,
  chunks and padding. Computed the first time a file is processed, outside the measured latency. For files
  processed before tracing: `python -m scripts.extract_features --dataset librispeech-test-clean`.

After pulling: `alembic upgrade head` (migration 0006), then restart the API and the worker. Runs made before
this have no stage events; the dashboard shows them with coarse timings (prepare / inference / scoring).
The dashboard shows transcripts: set `DASHBOARD_TOKEN` (and open `/dashboard?token=...`) or
`DASHBOARD_ENABLED=false` on any machine others can reach.

## Quick API check

    curl -H "X-API-Key: $KEY" -F file=@sample.wav -F model_version=faster-whisper-tiny localhost:8000/audio
    curl -H "X-API-Key: $KEY" localhost:8000/runs/<run_id>
    curl -H "X-API-Key: $KEY" localhost:8000/datasets/librispeech-test-clean/leaderboard

## Test

    pytest          # no model, torch or S3 server needed; S3 tests skip without one

## Layout

    app/api           routes, auth dependency, schemas (never touches storage/DB directly)
    app/services      submit_audio / get_status / get_result
    app/storage       StorageBackend interface: local disk and S3/MinIO
    app/db            SQLAlchemy models, repositories (status state machine, atomic claim)
    app/pipeline      ModelWrapper interface, stub and faster-whisper models, registry, Silero VAD,
                      derived cache, worker (with per-run profile), queues
    app/evaluation    WER/CER with versioned normalization (whisper_norm/ is vendored), leaderboard, summaries
    scripts           import, baseline, report, evaluate, export, compare, extract_features, rotate_keys, recovery
    app/dashboard     dashboard page (static/index.html, ECharts vendored) and its read-only JSON API
    app/observability logging, /metrics, per-run tracing (tracing.py)
    migrations        Alembic (0001 to 0006)
    results           measured baselines, exports, comparisons (see results/README.md)

## Decisions made while implementing

- `list` on the storage interface is named `list_keys` (avoids shadowing the builtin).
- Duplicate uploads: every upload gets its own blob and audio row; `checksum` is indexed, not unique.
- Accepted formats: .wav .flac .ogg .mp3, validated by decoding the header with soundfile.
- Workers claim runs with an atomic `UPDATE ... WHERE status='queued'`, so duplicate or retried jobs are no-ops.
- `keys.json`, `.env` and `.env.gpu` are never committed (they hold API keys and database URLs).

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
