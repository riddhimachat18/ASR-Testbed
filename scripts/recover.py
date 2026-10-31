"""Run one recovery sweep by hand (works with QUEUE_BACKEND=celery).

    python -m scripts.recover                          # normal thresholds
    python -m scripts.recover --stuck-after 0 --queued-after 0   # force: treat everything as stale
"""
import argparse

from app.config import get_settings
from app.pipeline.queues import build_queue
from app.pipeline.recovery import RecoveryConfig, recover_runs
from app.runtime import build_runtime

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stuck-after", type=int)
    ap.add_argument("--queued-after", type=int)
    a = ap.parse_args()
    st = get_settings()
    cfg = RecoveryConfig.from_settings(st)
    if a.stuck_after is not None: cfg.stuck_after_s = a.stuck_after
    if a.queued_after is not None: cfg.queued_after_s = a.queued_after
    rt = build_runtime(st)
    print(recover_runs(rt.session_factory, build_queue(st, rt), cfg))
