"""Score finished runs against reference transcripts (backfill, or recompute with another normalization).

    python -m scripts.evaluate                                     # runs with a reference but no score yet
    python -m scripts.evaluate --recompute                         # re-score everything with NORMALIZER
    python -m scripts.evaluate --recompute --normalization whisper-en-v1 \\
        --dataset librispeech-test-clean --model faster-whisper-tiny --config-hash <hash or prefix>

Re-scoring OVERWRITES wer, cer and metrics (profile and transcript stay). To keep the old numbers, export the
runs first: python -m scripts.export_runs ... (the export records the normalization it was scored under).
"""
import argparse

from sqlalchemy import select

from app.config import get_settings
from app.db.models import Audio, Dataset, ReferenceTranscript, RunStatus, TranscriptionRun
from app.db.session import make_engine, make_session_factory
from app.evaluation.metrics import get_normalizer
from app.evaluation.service import evaluate_run

BATCH = 500

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true", help="also re-score runs that already have a score")
    ap.add_argument("--normalization", help="default: the NORMALIZER setting")
    ap.add_argument("--dataset", help="only runs on this dataset's audio")
    ap.add_argument("--model", help="only this model_version")
    ap.add_argument("--config-hash", help="only this config hash (a prefix is enough)")
    a = ap.parse_args()
    st = get_settings()
    normalization = a.normalization or st.normalizer
    get_normalizer(normalization)  # fail before touching anything if misspelled
    sf = make_session_factory(make_engine(st.database_url, st.db_pool_size, st.db_max_overflow, st.db_pool_timeout))
    q = (select(TranscriptionRun.id)
         .join(ReferenceTranscript, ReferenceTranscript.audio_id == TranscriptionRun.audio_id)
         .where(TranscriptionRun.status == RunStatus.done).order_by(TranscriptionRun.id))
    if not a.recompute:
        q = q.where(TranscriptionRun.wer.is_(None), TranscriptionRun.metrics.is_(None))
    if a.dataset:
        q = q.join(Audio, Audio.id == TranscriptionRun.audio_id).join(Dataset, Dataset.id == Audio.dataset_id) \
             .where(Dataset.name == a.dataset)
    if a.model:
        q = q.where(TranscriptionRun.model_version == a.model)
    if a.config_hash:
        q = q.where(TranscriptionRun.config_hash.startswith(a.config_hash))
    with sf() as s:
        ids = list(s.scalars(q))
    scored = 0
    for i in range(0, len(ids), BATCH):
        with sf() as s:
            scored += sum(evaluate_run(s, rid, normalization) for rid in ids[i:i + BATCH])
            s.commit()
    print(f"scored {scored} of {len(ids)} candidate runs with {normalization}")
