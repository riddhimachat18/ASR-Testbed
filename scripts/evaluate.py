"""Score finished runs against reference transcripts (backfill / recompute).

    python -m scripts.evaluate                 # runs that have a reference but no score yet
    python -m scripts.evaluate --recompute     # re-score everything (e.g. after changing the normalisation)
"""
import argparse

from sqlalchemy import select

from app.config import get_settings
from app.db.models import ReferenceTranscript, RunStatus, TranscriptionRun
from app.db.session import make_engine, make_session_factory
from app.evaluation.service import evaluate_run

BATCH = 500

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    st = get_settings()
    sf = make_session_factory(make_engine(st.database_url, st.db_pool_size, st.db_max_overflow, st.db_pool_timeout))
    q = (select(TranscriptionRun.id)
         .join(ReferenceTranscript, ReferenceTranscript.audio_id == TranscriptionRun.audio_id)
         .where(TranscriptionRun.status == RunStatus.done).order_by(TranscriptionRun.id))
    if not a.recompute:
        q = q.where(TranscriptionRun.wer.is_(None), TranscriptionRun.metrics.is_(None))
    with sf() as s:
        ids = list(s.scalars(q))
    scored = 0
    for i in range(0, len(ids), BATCH):
        with sf() as s:
            scored += sum(evaluate_run(s, rid) for rid in ids[i:i + BATCH])
            s.commit()
    print(f"scored {scored} of {len(ids)} candidate runs")
