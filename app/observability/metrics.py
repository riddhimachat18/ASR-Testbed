"""Prometheus text exposition of queue depth and run status, computed from the database (the one source of truth
shared by the API, every Celery worker and the sweeper)."""
from datetime import datetime, timezone

from sqlalchemy import func, select

from app.db.models import RunStatus, TranscriptionRun as Run


def _esc(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _age(ts: datetime | None, now: datetime) -> float:
    if ts is None:
        return 0.0
    if ts.tzinfo is None:  # SQLite returns naive UTC
        ts = ts.replace(tzinfo=timezone.utc)
    return max((now - ts).total_seconds(), 0.0)


def collect_metrics(session_factory, queue, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    with session_factory() as s:
        rows = s.execute(select(Run.model_version, Run.status, func.count()).group_by(Run.model_version, Run.status)).all()
        oldest_queued = s.scalar(select(func.min(Run.created_at)).where(Run.status == RunStatus.queued))
        oldest_processing = s.scalar(select(func.min(Run.started_at)).where(Run.status == RunStatus.processing))
        s.rollback()  # read-only: release the connection immediately

    by_status = {st.value: 0 for st in RunStatus}
    by_model: dict[tuple[str, str], int] = {}
    for model, status, n in rows:
        by_status[status.value] += n
        by_model[(model, status.value)] = n

    out: list[str] = []

    def metric(name: str, help_text: str, samples: list[tuple[dict, float]]):
        out.append(f"# HELP {name} {help_text}")
        out.append(f"# TYPE {name} gauge")
        for labels, value in samples:
            lab = ",".join(f'{k}="{_esc(str(v))}"' for k, v in labels.items())
            out.append(f"{name}{{{lab}}} {value}" if lab else f"{name} {value}")

    metric("asr_runs", "Transcription runs by status.", [({"status": k}, v) for k, v in by_status.items()])
    metric("asr_runs_by_model", "Transcription runs by model version and status.",
           [({"model_version": m, "status": st}, n) for (m, st), n in sorted(by_model.items())])
    metric("asr_oldest_queued_age_seconds", "Age of the oldest run still queued (0 if none).",
           [({}, round(_age(oldest_queued, now), 3))])
    metric("asr_oldest_processing_age_seconds", "Time the longest-running run has been processing (0 if none).",
           [({}, round(_age(oldest_processing, now), 3))])
    for name, value in sorted(queue.stats().items()):
        metric(f"asr_{name}", f"Queue backend statistic: {name}.", [({}, value)])
    return "\n".join(out) + "\n"
