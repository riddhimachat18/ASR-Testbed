"""Import a LibriSpeech split (test-clean, test-other, ...) into the ASR testbed.

Usage
─────
    python scripts/import_librispeech.py --source LibriSpeech/test-clean --dry-run
    python scripts/import_librispeech.py --source LibriSpeech/test-clean
    python scripts/import_librispeech.py --source LibriSpeech/test-other            # dataset librispeech-test-other
    python scripts/import_librispeech.py --source LibriSpeech/test-clean --model faster-whisper-tiny --enqueue
    python scripts/import_librispeech.py --source LibriSpeech/test-clean --limit 50

What it does
────────────
1. Walks <source>/<speaker>/<chapter>/ for FLAC files and their .trans.txt references.
2. Converts each FLAC to WAV in memory, uploads it to the configured storage, and inserts Audio (with
   external_id = the LibriSpeech utterance ID, e.g. 1089-134686-0000) and ReferenceTranscript rows, tagged with
   the dataset (default: librispeech-<split folder name>).
3. Files already present are skipped, so the script is safe to re-run. A re-run also BACKFILLS external_id on
   rows imported before Part H (migration 0005): a file is matched by checksum, or, if the WAV encoder output
   changed since, by reference text and duration within the dataset.
4. With --model, a run is created per newly imported file; with --enqueue those runs are also sent to Celery.

Requirements: run from the project root; the API need not be running; ffmpeg/libsndfile for FLAC decoding.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # make `app` importable when run from the project root

import soundfile as sf
from sqlalchemy import select

from app.config import get_settings
from app.db.models import Audio, Dataset, ReferenceTranscript, User
from app.db.repositories import RunRepo
from app.db.session import make_engine, make_session_factory
from app.pipeline.registry import ModelRegistry
from app.pipeline.run_config import resolve_run_config
from app.storage.factory import get_storage
from app.storage.keys import new_audio_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("import_librispeech")

REFERENCE_LANGUAGE = "en"
DURATION_TOLERANCE_S = 0.002  # for the backfill fallback match


def parse_trans_file(path: Path) -> dict[str, str]:
    """{utterance_id: reference_text} from a .trans.txt file."""
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            result[parts[0]] = parts[1]
    return result


def flac_to_wav_bytes(flac_path: Path) -> tuple[bytes, int, float]:
    """(wav_bytes, sample_rate, duration_s). Keeps the original rate; resampling is done by DerivedCache."""
    with sf.SoundFile(str(flac_path)) as src:
        sample_rate = src.samplerate
        data = src.read(dtype="float32", always_2d=False)
    buf = io.BytesIO()
    with sf.SoundFile(buf, mode="w", samplerate=sample_rate, channels=1 if data.ndim == 1 else data.shape[1],
                      subtype="PCM_16", format="WAV") as out:
        out.write(data)
    return buf.getvalue(), sample_rate, len(data) / sample_rate


def find_flac_files(source: Path) -> list[tuple[Path, str]]:
    """(flac_path, utterance_id) for every utterance listed in a .trans.txt under source."""
    pairs = []
    for trans_file in sorted(source.rglob("*.trans.txt")):
        for utt_id in parse_trans_file(trans_file):
            flac = trans_file.parent / f"{utt_id}.flac"
            if flac.exists():
                pairs.append((flac, utt_id))
            else:
                log.warning("missing FLAC for utterance %s", utt_id)
    return pairs


def default_dataset_name(source: Path) -> str:
    return f"librispeech-{source.resolve().name}"  # LibriSpeech/test-clean -> librispeech-test-clean


def get_or_create_dataset(s, user_id: str, name: str) -> Dataset:
    ds = s.scalar(select(Dataset).where(Dataset.user_id == user_id, Dataset.name == name))
    if ds is None:
        ds = Dataset(user_id=user_id, name=name)
        s.add(ds)
        s.flush()
    return ds


def find_existing(s, dataset_id: str, checksum: str, ref_text: str, duration: float) -> Audio | None:
    """The already-imported row for this file: by checksum (preferring this dataset), else, for rows without an
    external_id in this dataset, by identical reference text and duration."""
    rows = s.scalars(select(Audio).where(Audio.checksum == checksum)).all()
    for a in rows:
        if a.dataset_id == dataset_id:
            return a
    if rows:
        return rows[0]
    candidates = s.scalars(
        select(Audio).join(ReferenceTranscript, ReferenceTranscript.audio_id == Audio.id)
        .where(Audio.dataset_id == dataset_id, Audio.external_id.is_(None), ReferenceTranscript.text == ref_text)
    ).all()
    matches = [a for a in candidates if abs(a.duration - duration) <= DURATION_TOLERANCE_S]
    return matches[0] if len(matches) == 1 else None


def import_split(session_factory, storage, source: Path, dataset_name: str, user_id: str, *,
                 model: str | None = None, run_config: tuple[dict, str] | None = None,
                 limit: int | None = None) -> dict:
    """Import (or backfill) every utterance under source. Returns counts and the ids of runs created."""
    pairs = find_flac_files(source)[:limit] if limit else find_flac_files(source)
    references: dict[str, str] = {}
    for trans_file in source.rglob("*.trans.txt"):
        references.update(parse_trans_file(trans_file))
    with session_factory() as s:
        dataset_id = get_or_create_dataset(s, user_id, dataset_name).id
        s.commit()

    counts = {"imported": 0, "skipped": 0, "backfilled": 0, "failed": 0}
    run_ids: list[str] = []
    t0 = time.perf_counter()
    for i, (flac_path, utt_id) in enumerate(pairs, 1):
        ref_text = references[utt_id]
        try:
            wav, sample_rate, duration = flac_to_wav_bytes(flac_path)
        except Exception as e:  # noqa: BLE001
            log.error("failed to read %s: %s", flac_path, e)
            counts["failed"] += 1
            continue
        checksum = hashlib.sha256(wav).hexdigest()
        with session_factory() as s:
            existing = find_existing(s, dataset_id, checksum, ref_text, duration)
            if existing is not None:
                if existing.dataset_id is None:
                    existing.dataset_id = dataset_id
                if existing.dataset_id == dataset_id and existing.external_id is None:
                    taken = s.scalar(select(Audio.id).where(Audio.dataset_id == dataset_id,
                                                            Audio.external_id == utt_id))
                    if taken is None:
                        existing.external_id = utt_id
                        counts["backfilled"] += 1
                if s.scalar(select(ReferenceTranscript).where(ReferenceTranscript.audio_id == existing.id)) is None:
                    s.add(ReferenceTranscript(audio_id=existing.id, text=ref_text, language=REFERENCE_LANGUAGE,
                                              source=dataset_name))
                s.commit()
                counts["skipped"] += 1
            else:
                key = new_audio_key(".wav")
                try:
                    blob = storage.save_audio(key, io.BytesIO(wav))
                except Exception as e:  # noqa: BLE001
                    log.error("storage upload failed for %s: %s", utt_id, e)
                    counts["failed"] += 1
                    continue
                audio = Audio(user_id=user_id, storage_key=key, checksum=blob.sha256, duration=duration,
                              original_sample_rate=sample_rate, size_bytes=blob.size_bytes,
                              dataset_id=dataset_id, external_id=utt_id)
                s.add(audio)
                s.flush()
                s.add(ReferenceTranscript(audio_id=audio.id, text=ref_text, language=REFERENCE_LANGUAGE,
                                          source=dataset_name))
                if model and run_config:
                    run_ids.append(RunRepo(s).add(audio.id, model, run_config[0], config_hash=run_config[1]).id)
                s.commit()
                counts["imported"] += 1
        if i % 200 == 0:
            rate = i / (time.perf_counter() - t0)
            log.info("progress %d/%d %s  %.1f files/s", i, len(pairs), counts, rate)
    counts["seconds"] = round(time.perf_counter() - t0, 1)
    counts["run_ids"] = run_ids
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Import a LibriSpeech split (and backfill utterance IDs)")
    ap.add_argument("--source", default="LibriSpeech/test-clean", help="split directory")
    ap.add_argument("--dataset-name", default=None, help="default: librispeech-<split folder name>")
    ap.add_argument("--model", default=None, help="create a run per newly imported file with this model")
    ap.add_argument("--config", default="{}", help="run config JSON for --model (validated and hashed)")
    ap.add_argument("--enqueue", action="store_true", help="send the created runs to Celery")
    ap.add_argument("--dry-run", action="store_true", help="list what would be imported, change nothing")
    ap.add_argument("--limit", type=int, default=None, help="only the first N utterances")
    ap.add_argument("--user", default=None, help="owner username (default: first user in the DB)")
    a = ap.parse_args()

    source = Path(a.source)
    if not source.exists():
        sys.exit(f"source path does not exist: {source}")
    dataset_name = a.dataset_name or default_dataset_name(source)
    if a.dry_run:
        pairs = find_flac_files(source)
        log.info("[dry-run] %d utterances under %s -> dataset %s", len(pairs), source, dataset_name)
        return

    settings = get_settings()
    sf_ = make_session_factory(make_engine(settings.database_url))
    with sf_() as s:
        q = select(User).where(User.username == a.user) if a.user else select(User).order_by(User.created_at)
        user = s.scalars(q).first()
        if user is None:
            sys.exit(f"user {a.user!r} not found" if a.user else "no users; run: python -m scripts.create_user <name>")
        user_id = user.id
    run_config = None
    if a.model:
        registry = ModelRegistry(settings)
        if a.model not in registry.names():
            sys.exit(f"unknown model {a.model!r}; available: {registry.names()}")
        run_config = resolve_run_config(a.model, json.loads(a.config), registry.options_model(a.model))

    counts = import_split(sf_, get_storage(settings), source, dataset_name, user_id,
                          model=a.model, run_config=run_config, limit=a.limit)
    run_ids = counts.pop("run_ids")
    log.info("done: dataset=%s %s", dataset_name, counts)
    if run_ids and a.enqueue:
        from app.celery_app import process_run
        for rid in run_ids:
            process_run.delay(rid)
        log.info("enqueued %d runs", len(run_ids))
    elif run_ids:
        log.info("%d runs created but not enqueued (use --enqueue, or the recovery sweep republishes them)",
                 len(run_ids))


if __name__ == "__main__":
    main()
