"""Compute audio features (app/pipeline/audio_features.py) for audio that has none yet, e.g. files processed
before the features existed. Workers compute them automatically for every new file.

    python -m scripts.extract_features                                   # every audio file without features
    python -m scripts.extract_features --dataset librispeech-test-clean --recompute

Uses the default run config's chunk plan (VAD, 30 s), read from the derived cache when present, so on
test-clean it reuses the VAD results of the baseline runs instead of running VAD again.
"""
import argparse
import time

from sqlalchemy import select

from app.config import get_settings
from app.db.models import Audio, AudioFeatures, Dataset
from app.db.repositories import AudioFeaturesRepo
from app.db.session import make_engine, make_session_factory
from app.pipeline.audio_features import FEATURES_VERSION, features_from_wav
from app.pipeline.prepare import AudioPreparer
from app.pipeline.run_config import RunConfig
from app.storage.factory import get_storage

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="only this dataset's audio")
    ap.add_argument("--recompute", action="store_true", help="also files that already have features")
    ap.add_argument("--config", help="RunConfig JSON for the chunk plan (default: the default config)")
    a = ap.parse_args()
    st = get_settings()
    sf = make_session_factory(make_engine(st.database_url, st.db_pool_size, st.db_max_overflow, st.db_pool_timeout))
    preparer = AudioPreparer(get_storage(st), vad_device=st.vad_device)
    cfg = RunConfig.model_validate_json(a.config) if a.config else RunConfig()
    q = select(Audio.id, Audio.storage_key, Audio.checksum).order_by(Audio.id)
    if a.dataset:
        q = q.join(Dataset, Dataset.id == Audio.dataset_id).where(Dataset.name == a.dataset)
    if not a.recompute:
        q = q.outerjoin(AudioFeatures, AudioFeatures.audio_id == Audio.id).where(
            (AudioFeatures.audio_id.is_(None)) | (AudioFeatures.version != FEATURES_VERSION))
    with sf() as s:
        todo = list(s.execute(q))
    started, done = time.perf_counter(), 0
    for audio_id, key, checksum in todo:
        stream, prepared = preparer.prepare(key, checksum, cfg)
        with stream:
            t = time.perf_counter()
            feats = features_from_wav(stream, [(c.start, c.end) for c in prepared.chunks],
                                      vad=cfg.chunking.strategy == "vad")
            feats["extra"]["compute_ms"] = round((time.perf_counter() - t) * 1000.0, 2)
        with sf() as s:
            AudioFeaturesRepo(s).upsert(audio_id, feats)
            s.commit()
        done += 1
        if done % 200 == 0:
            print(f"{done}/{len(todo)} ({time.perf_counter() - started:.0f} s)", flush=True)
    print(f"features computed for {done} audio files in {time.perf_counter() - started:.1f} s")
