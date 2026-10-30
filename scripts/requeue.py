"""Recovery sweep: reset runs stuck in 'processing' back to 'queued', then process every queued run.

Only run this when NO server/worker is running (it checks for that).
Usage:  python -m scripts.requeue
Tip:    STUB_SLEEP_SECONDS=0.05 WORKER_THREADS=8 python -m scripts.requeue
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import func, select, update

from app.config import get_settings
from app.db.models import RunStatus, TranscriptionRun
from app.db.session import make_engine, make_session_factory
from app.pipeline.registry import ModelRegistry
from app.pipeline.worker import RunProcessor
from app.storage.factory import get_storage


def _done_count(sf) -> int:
    with sf() as s:
        return s.scalar(select(func.count()).select_from(TranscriptionRun)
                        .where(TranscriptionRun.status == RunStatus.done))


def main() -> None:
    st = get_settings()
    engine = make_engine(st.database_url, st.db_pool_size, st.db_max_overflow, st.db_pool_timeout)
    sf = make_session_factory(engine)

    # Safety check: if runs are still completing, a server is alive. Resetting 'processing'
    # runs now would make two workers process the same run.
    before = _done_count(sf)
    time.sleep(4)
    after = _done_count(sf)
    if after != before:
        sys.exit(f"Runs are still completing (done {before} -> {after}): a server is still running.\n"
                 "Stop it first (find it with: lsof -i :8000), then rerun this script.")

    with sf() as s:
        reset = s.execute(
            update(TranscriptionRun).where(TranscriptionRun.status == RunStatus.processing)
            .values(status=RunStatus.queued)
        ).rowcount
        s.commit()
        ids = list(s.scalars(
            select(TranscriptionRun.id).where(TranscriptionRun.status == RunStatus.queued)
            .order_by(TranscriptionRun.created_at)
        ))
    print(f"reset {reset} stuck 'processing' runs; processing {len(ids)} queued runs with {st.worker_threads} threads")

    proc = RunProcessor(sf, get_storage(st), ModelRegistry(st))

    def safe_process(run_id: str) -> None:
        try:
            proc.process(run_id)
        except Exception as e:  # noqa: BLE001  one bad run must not abort the sweep
            print(f"  run {run_id} raised {e!r}; continuing")

    start, n = time.time(), 0
    with ThreadPoolExecutor(st.worker_threads) as pool:
        for _ in pool.map(safe_process, ids):
            n += 1
            if n % 1000 == 0:
                print(f"  {n}/{len(ids)}  ({n / (time.time() - start):.0f} runs/s)")
    print(f"finished in {time.time() - start:.0f}s. Final status counts:")
    with sf() as s:
        for status, c in s.execute(select(TranscriptionRun.status, func.count()).group_by(TranscriptionRun.status)):
            print(f"  {status.value:<11}{c}")


if __name__ == "__main__":
    main()