"""Baseline runner for LibriSpeech test-clean.

What it does
────────────
1. Finds every Audio row in the 'librispeech-test-clean' dataset that has a
   ReferenceTranscript but no completed run for the given model.
2. Creates a TranscriptionRun for each (if not already queued/processing/done).
3. Polls until all runs finish (done or failed).
4. Writes aggregate metrics to results/baseline_<timestamp>.json.

Usage
─────
    # queue runs and wait (will take a while on CPU)
    python scripts/run_baseline.py --model faster-whisper-tiny

    # skip already-done runs and just wait for the in-flight ones
    python scripts/run_baseline.py --model faster-whisper-tiny --no-submit

    # limit to first 50 files (quick smoke test)
    python scripts/run_baseline.py --model faster-whisper-tiny --limit 50

Requirements
────────────
* The API server + Celery worker must be running:
      uvicorn app.main:factory --factory
      celery -A app.celery_app worker --concurrency=2 --loglevel=info
* Run from the project root.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
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
from app.pipeline.registry import ModelRegistry
from app.pipeline.run_config import resolve_run_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_baseline")

DATASET_NAME = "librispeech-test-clean"
RESULTS_DIR = Path("results")
POLL_INTERVAL_S = 10


def submit_runs(session_factory, model: str, config_resolved: dict, config_hash: str,
                limit: int | None) -> list[str]:
    """Create TranscriptionRun rows for all unprocessed files and push them to the Celery queue.
    Returns list of run IDs."""
    from app.celery_app import process_run  # import here so celery_app connects to Redis

    run_ids: list[str] = []

    with session_factory() as s:
        ds = s.scalar(select(Dataset).where(Dataset.name == DATASET_NAME))
        if ds is None:
            log.error("Dataset '%s' not found. Run import_librispeech.py first.", DATASET_NAME)
            sys.exit(1)

        audio_ids: list[str] = list(s.scalars(
            select(Audio.id)
            .join(ReferenceTranscript, ReferenceTranscript.audio_id == Audio.id)
            .where(Audio.dataset_id == ds.id)
        ).all())
        log.info("Found %d audio files with references in dataset", len(audio_ids))
        if limit:
            audio_ids = audio_ids[:limit]
            log.info("Limited to %d files", limit)

        repo = RunRepo(s)
        submitted = 0
        reused = 0
        new_run_ids: list[str] = []

        for audio_id in audio_ids:
            # Check if there's already a done run for this model+config
            existing = s.scalar(
                select(TranscriptionRun)
                .where(
                    TranscriptionRun.audio_id == audio_id,
                    TranscriptionRun.model_version == model,
                    TranscriptionRun.config_hash == config_hash,
                    TranscriptionRun.status.in_([
                        RunStatus.queued, RunStatus.processing, RunStatus.done
                    ]),
                )
            )
            if existing is not None:
                run_ids.append(existing.id)
                reused += 1
                continue
            run = repo.add(audio_id, model, config_resolved, config_hash=config_hash)
            run_ids.append(run.id)
            new_run_ids.append(run.id)
            submitted += 1

        s.commit()
        log.info("Submitted %d new runs, reused %d existing", submitted, reused)

    # Push new runs into Redis AFTER committing so the worker can see the rows
    if new_run_ids:
        log.info("Enqueueing %d runs to Celery/Redis...", len(new_run_ids))
        for run_id in new_run_ids:
            process_run.delay(run_id)
        log.info("Done enqueueing.")

    return run_ids


def wait_for_runs(session_factory, run_ids: list[str]) -> None:
    """Block until all run_ids are in a terminal state (done or failed)."""
    pending = set(run_ids)
    total = len(pending)
    log.info("Waiting for %d runs to complete...", total)

    while pending:
        with session_factory() as s:
            rows = s.execute(
                select(TranscriptionRun.id, TranscriptionRun.status)
                .where(TranscriptionRun.id.in_(list(pending)))
            ).all()

        done_now = {r.id for r in rows if r.status in (RunStatus.done, RunStatus.failed)}
        pending -= done_now

        completed = total - len(pending)
        log.info("  %d / %d complete (%d pending)", completed, total, len(pending))

        if not pending:
            break
        time.sleep(POLL_INTERVAL_S)


def compute_summary(session_factory, run_ids: list[str], model: str) -> dict:
    """Aggregate metrics from all completed runs."""
    with session_factory() as s:
        rows = s.execute(
            select(
                TranscriptionRun.id,
                TranscriptionRun.status,
                TranscriptionRun.wer,
                TranscriptionRun.cer,
                TranscriptionRun.latency,
                TranscriptionRun.metrics,
                TranscriptionRun.audio_id,
            ).where(TranscriptionRun.id.in_(run_ids))
        ).all()

        audio_durations = {}
        audio_ids = [r.audio_id for r in rows]
        if audio_ids:
            for a in s.execute(select(Audio.id, Audio.duration).where(Audio.id.in_(audio_ids))).all():
                audio_durations[a.id] = a.duration

    done = [r for r in rows if r.status == RunStatus.done and r.wer is not None]
    failed = [r for r in rows if r.status == RunStatus.failed]
    unscored = [r for r in rows if r.status == RunStatus.done and r.wer is None]

    if not done:
        log.warning("No scored runs found.")
        return {"model": model, "runs_done": 0, "runs_failed": len(failed)}

    # Micro-averaged WER (correct way: total errors / total reference words)
    total_ref_words = sum(r.metrics["ref_words"] for r in done if r.metrics and "ref_words" in r.metrics)
    total_word_errors = sum(
        (r.metrics.get("word_substitutions", 0) + r.metrics.get("word_deletions", 0)
         + r.metrics.get("word_insertions", 0))
        for r in done if r.metrics
    )
    micro_wer = total_word_errors / total_ref_words if total_ref_words > 0 else None

    total_ref_chars = sum(r.metrics["ref_chars"] for r in done if r.metrics and "ref_chars" in r.metrics)
    total_char_errors = sum(
        (r.metrics.get("char_substitutions", 0) + r.metrics.get("char_deletions", 0)
         + r.metrics.get("char_insertions", 0))
        for r in done if r.metrics
    )
    micro_cer = total_char_errors / total_ref_chars if total_ref_chars > 0 else None

    # Per-file WER distribution
    per_file_wers = sorted(r.wer for r in done if r.wer is not None)
    n = len(per_file_wers)

    def pct(p: float) -> float | None:
        if not per_file_wers:
            return None
        idx = min(int(p * n), n - 1)
        return round(per_file_wers[idx], 4)

    # Latency + real-time factor
    latencies = [r.latency for r in done if r.latency is not None]
    rtfs = []
    for r in done:
        dur = audio_durations.get(r.audio_id)
        if r.latency and dur and dur > 0:
            rtfs.append(r.latency / dur)

    latencies_sorted = sorted(latencies)
    rtfs_sorted = sorted(rtfs)

    def p(vals: list[float], pct_: float) -> float | None:
        if not vals:
            return None
        return round(vals[min(int(pct_ * len(vals)), len(vals) - 1)], 4)

    return {
        "model": model,
        "dataset": DATASET_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runs_total": len(rows),
        "runs_done": len(done),
        "runs_failed": len(failed),
        "runs_unscored": len(unscored),
        "wer": {
            "micro": round(micro_wer, 4) if micro_wer is not None else None,
            "per_file_mean": round(sum(per_file_wers) / n, 4) if n else None,
            "p10": pct(0.10),
            "p25": pct(0.25),
            "p50": pct(0.50),
            "p75": pct(0.75),
            "p90": pct(0.90),
            "p95": pct(0.95),
        },
        "cer": {
            "micro": round(micro_cer, 4) if micro_cer is not None else None,
        },
        "latency_s": {
            "mean": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "p50": p(latencies_sorted, 0.50),
            "p95": p(latencies_sorted, 0.95),
        },
        "rtf": {
            "mean": round(sum(rtfs) / len(rtfs), 3) if rtfs else None,
            "p50": p(rtfs_sorted, 0.50),
            "p95": p(rtfs_sorted, 0.95),
            "note": "rtf < 1.0 means faster than real-time",
        },
        "total_ref_words": total_ref_words,
        "total_word_errors": total_word_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline evaluation on LibriSpeech test-clean")
    parser.add_argument("--model", default="faster-whisper-tiny",
                        help="Model version to benchmark (default: faster-whisper-tiny)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N files")
    parser.add_argument("--no-submit", action="store_true",
                        help="Skip submission; only wait for existing runs")
    parser.add_argument("--no-wait", action="store_true",
                        help="Submit runs but do not wait for completion (exit immediately)")
    args = parser.parse_args()

    settings = get_settings()
    engine = make_engine(settings.database_url)
    session_factory = make_session_factory(engine)
    registry = ModelRegistry(settings)

    if args.model not in registry.names():
        log.error("Unknown model '%s'. Available: %s", args.model, registry.names())
        sys.exit(1)

    config_resolved, config_hash = resolve_run_config(
        args.model, {}, registry.options_model(args.model)
    )

    if not args.no_submit:
        run_ids = submit_runs(session_factory, args.model, config_resolved, config_hash, args.limit)
    else:
        # Fetch existing run IDs for this model/config
        from app.celery_app import process_run
        with session_factory() as s:
            ds = s.scalar(select(Dataset).where(Dataset.name == DATASET_NAME))
            if ds is None:
                log.error("Dataset not found.")
                sys.exit(1)
            rows = s.execute(
                select(TranscriptionRun.id, TranscriptionRun.status)
                .join(Audio, Audio.id == TranscriptionRun.audio_id)
                .where(
                    Audio.dataset_id == ds.id,
                    TranscriptionRun.model_version == args.model,
                    TranscriptionRun.config_hash == config_hash,
                )
            ).all()
        run_ids = [r.id for r in rows]
        # Re-enqueue any that are still queued but not in Redis (e.g. after a restart)
        still_queued = [r.id for r in rows if r.status == RunStatus.queued]
        if still_queued:
            log.info("Re-enqueueing %d runs still in 'queued' state...", len(still_queued))
            for run_id in still_queued:
                process_run.delay(run_id)
        log.info("Found %d existing runs (%d re-enqueued)", len(run_ids), len(still_queued))

    if args.no_wait:
        log.info("Runs submitted. Re-run without --no-wait to wait and report results.")
        return

    wait_for_runs(session_factory, run_ids)

    summary = compute_summary(session_factory, run_ids, args.model)

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out_path = RESULTS_DIR / f"baseline_{args.model.replace('/', '_')}_{ts}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    log.info("Results written to %s", out_path)

    # Print a short summary to stdout
    print("\n" + "═" * 60)
    print(f"  Baseline: {args.model} on {DATASET_NAME}")
    print("═" * 60)
    w = summary["wer"]
    c = summary["cer"]
    lat = summary["latency_s"]
    rtf = summary["rtf"]
    print(f"  Runs:     {summary['runs_done']} done, {summary['runs_failed']} failed")
    print(f"  WER:      {w['micro']:.1%}  (micro-avg)   p50={w['p50']:.1%}  p95={w['p95']:.1%}")
    print(f"  CER:      {c['micro']:.1%}  (micro-avg)")
    print(f"  Latency:  mean={lat['mean']:.2f}s  p50={lat['p50']:.2f}s  p95={lat['p95']:.2f}s")
    print(f"  RTF:      mean={rtf['mean']:.3f}  p50={rtf['p50']:.3f}  p95={rtf['p95']:.3f}")
    print(f"            ({rtf['note']})")
    print("═" * 60)
    print(f"  Saved to: {out_path}")
    print()


if __name__ == "__main__":
    main()
