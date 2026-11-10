"""Run one model + settings over a whole dataset and write a baseline summary JSON. CPU and GPU alike.

    # CPU, single worker (the fair speed measurement):
    FORKED_BY_MULTIPROCESSING=1 celery -A app.celery_app worker --concurrency=1      # macOS
    python scripts/run_baseline.py --model faster-whisper-tiny --worker-concurrency 1

    # GPU machine (ASR_ENV_FILE=.env.gpu with DEVICE=cuda, COMPUTE_TYPE=float16):
    celery -A app.celery_app worker --pool=solo --concurrency=1                      # Windows
    python scripts/run_baseline.py --model faster-whisper-small --worker-concurrency 1
    python scripts/run_baseline.py --model faster-whisper-tiny --compute-type int8_float16 --worker-concurrency 1

    # 20-file smoke test; wait for runs already submitted; submit only
    python scripts/run_baseline.py --model faster-whisper-tiny --limit 20 --worker-concurrency 1
    python scripts/run_baseline.py --model faster-whisper-tiny --no-submit --worker-concurrency 1
    python scripts/run_baseline.py --model faster-whisper-tiny --no-wait

What it does
  1. Resolves the run config: --config JSON, overridden by --device/--compute-type/--beam-size/--temperature;
     unset hardware options take this machine's DEVICE/COMPUTE_TYPE. The config hash identifies the run set.
  2. Creates a run for every file of the dataset that has a reference and no queued/processing/done run with
     the same model and config hash, and enqueues it to Celery after committing.
  3. Waits until every run is done or failed, then writes results/<CPU|GPU>/baseline_<model>_<device>-<compute>
     _<timestamp>.json (see app/evaluation/summary.py and results/README.md for the fields).

--worker-concurrency is required when waiting: the script cannot see how many workers ran, and throughput and
latency mean nothing without it. Measure speed with one worker; more workers on one CPU measure contention.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select

from app.config import get_settings
from app.db.models import Audio, Dataset, ReferenceTranscript, RunStatus, TranscriptionRun
from app.db.repositories import RunRepo
from app.db.session import make_engine, make_session_factory
from app.evaluation.summary import git_commit, summarize
from app.pipeline.registry import ModelRegistry
from app.pipeline.run_config import ConfigInvalid, resolve_run_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_baseline")

POLL_INTERVAL_S = 10


def build_raw_config(a) -> dict:
    raw = json.loads(a.config)
    opts = raw.setdefault("model_options", {})
    dec = raw.setdefault("decoding", {})
    for key, val in (("device", a.device), ("compute_type", a.compute_type)):
        if val is not None:
            opts[key] = val
    for key, val in (("beam_size", a.beam_size), ("temperature", a.temperature)):
        if val is not None:
            dec[key] = val
    return raw


def dataset_audio_ids(s, dataset: str, limit: int | None) -> list[str]:
    ds = s.scalar(select(Dataset).where(Dataset.name == dataset))
    if ds is None:
        sys.exit(f"dataset {dataset!r} not found; import it first (scripts/import_librispeech.py)")
    ids = list(s.scalars(select(Audio.id).join(ReferenceTranscript, ReferenceTranscript.audio_id == Audio.id)
                         .where(Audio.dataset_id == ds.id).order_by(Audio.external_id, Audio.id)))
    return ids[:limit] if limit else ids


def submit_runs(sf, dataset: str, model: str, cfg: dict, cfg_hash: str, limit: int | None) -> list[str]:
    """Run ids covering the dataset for this model + config: existing live runs are reused, the rest created
    and enqueued."""
    from app.celery_app import process_run  # imported here so nothing connects to Redis before it is needed

    with sf() as s:
        audio_ids = dataset_audio_ids(s, dataset, limit)
        existing = {a: rid for a, rid in s.execute(
            select(TranscriptionRun.audio_id, TranscriptionRun.id).where(
                TranscriptionRun.audio_id.in_(audio_ids), TranscriptionRun.model_version == model,
                TranscriptionRun.config_hash == cfg_hash,
                TranscriptionRun.status.in_([RunStatus.queued, RunStatus.processing, RunStatus.done])))}
        new = [RunRepo(s).add(a, model, cfg, config_hash=cfg_hash).id for a in audio_ids if a not in existing]
        s.commit()
    for rid in new:  # after the commit, so a worker always finds the row
        process_run.delay(rid)
    log.info("%d files: %d runs reused, %d created and enqueued", len(audio_ids), len(existing), len(new))
    return list(existing.values()) + new


def existing_runs(sf, dataset: str, model: str, cfg_hash: str, limit: int | None) -> list[str]:
    """--no-submit: the runs already in the database; queued ones are re-enqueued (e.g. after a Redis restart)."""
    from app.celery_app import process_run

    with sf() as s:
        audio_ids = dataset_audio_ids(s, dataset, limit)
        rows = s.execute(select(TranscriptionRun.id, TranscriptionRun.status).where(
            TranscriptionRun.audio_id.in_(audio_ids), TranscriptionRun.model_version == model,
            TranscriptionRun.config_hash == cfg_hash)).all()
    for rid, status in rows:
        if status == RunStatus.queued:
            process_run.delay(rid)
    log.info("found %d existing runs (%d re-enqueued)", len(rows), sum(st == RunStatus.queued for _, st in rows))
    return [rid for rid, _ in rows]


def wait_for_runs(sf, run_ids: list[str]) -> None:
    pending = set(run_ids)
    while pending:
        with sf() as s:
            finished = set(s.scalars(select(TranscriptionRun.id).where(
                TranscriptionRun.id.in_(list(pending)),
                TranscriptionRun.status.in_([RunStatus.done, RunStatus.failed]))))
        pending -= finished
        log.info("%d / %d finished", len(run_ids) - len(pending), len(run_ids))
        if pending:
            time.sleep(POLL_INTERVAL_S)


def load_rows(sf, run_ids: list[str]) -> list[dict]:
    with sf() as s:
        rows = s.execute(
            select(TranscriptionRun.status, TranscriptionRun.wer, TranscriptionRun.metrics, TranscriptionRun.latency,
                   TranscriptionRun.profile, TranscriptionRun.started_at, Audio.duration)
            .join(Audio, Audio.id == TranscriptionRun.audio_id).where(TranscriptionRun.id.in_(run_ids))).all()
    return [{"status": r.status.value, "wer": r.wer, "metrics": r.metrics, "latency": r.latency,
             "profile": r.profile, "started_at": r.started_at, "duration": r.duration} for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline a model + settings over a dataset")
    ap.add_argument("--model", default="faster-whisper-tiny")
    ap.add_argument("--dataset", default="librispeech-test-clean")
    ap.add_argument("--device", choices=["cpu", "cuda"], help="default: this machine's DEVICE")
    ap.add_argument("--compute-type", help="default: this machine's COMPUTE_TYPE")
    ap.add_argument("--beam-size", type=int)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--config", default="{}", help="any other run config, as JSON")
    ap.add_argument("--worker-concurrency", type=int, help="Celery workers that processed the runs (required)")
    ap.add_argument("--results-dir", help="default: results/CPU or results/GPU by device")
    ap.add_argument("--limit", type=int, help="only the first N files (smoke test)")
    ap.add_argument("--no-submit", action="store_true", help="only wait for runs already in the database")
    ap.add_argument("--no-wait", action="store_true", help="submit and exit")
    a = ap.parse_args()
    if not a.no_wait and a.worker_concurrency is None:
        ap.error("--worker-concurrency is required (how many Celery workers process these runs?)")

    settings = get_settings()
    sf = make_session_factory(make_engine(settings.database_url))
    registry = ModelRegistry(settings)
    if a.model not in registry.names():
        sys.exit(f"unknown model {a.model!r}; available: {registry.names()}")
    try:
        cfg, cfg_hash = resolve_run_config(a.model, build_raw_config(a), registry.options_model(a.model))
    except ConfigInvalid as e:
        sys.exit(f"invalid config: {e.errors}")
    opts = cfg.get("model_options", {})
    log.info("model=%s config_hash=%s options=%s decoding=%s", a.model, cfg_hash[:12], opts, cfg["decoding"])

    run_ids = (existing_runs if a.no_submit else submit_runs)(sf, a.dataset, a.model, cfg, cfg_hash, a.limit)
    if a.no_wait:
        log.info("submitted; re-run with --no-submit to wait and report")
        return
    wait_for_runs(sf, run_ids)

    summary = {"model": a.model, "dataset": a.dataset, "timestamp": datetime.now(timezone.utc).isoformat(),
               "git_commit": git_commit(), "config_hash": cfg_hash, "config": cfg,
               "device": opts.get("device"), "compute_type": opts.get("compute_type"),
               "worker_concurrency": a.worker_concurrency, "limit": a.limit,
               **summarize(load_rows(sf, run_ids))}
    device = opts.get("device") or "cpu"
    out_dir = Path(a.results_dir or f"results/{'GPU' if device == 'cuda' else 'CPU'}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = out_dir / f"baseline_{a.model}_{device}-{opts.get('compute_type', 'na')}_{ts}.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    from scripts.report_baseline import print_report
    print_report(summary)
    print(f"  Saved to: {out}\n")


if __name__ == "__main__":
    main()
