"""Import LibriSpeech test-clean into the GPU stack (Postgres + GPU Celery task).

Identical to import_librispeech.py except:
  - Uses GPUSettings (reads .env.gpu → Postgres DATABASE_URL)
  - Enqueues via gpu.celery_app_gpu.process_run_gpu (task name asr.process_run_gpu)
  - Reads model registry from ModelRegistryGPU

Usage:
    python scripts/import_librispeech_gpu.py --source data/test-clean/LibriSpeech/test-clean --model faster-whisper-tiny
"""
from __future__ import annotations

import argparse
import hashlib
import io
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import soundfile as sf
from sqlalchemy import select

from gpu.config_gpu import get_gpu_settings
from gpu.pipeline.registry_gpu import ModelRegistryGPU
from app.db.models import Audio, Dataset, ReferenceTranscript
from app.db.repositories import RunRepo
from app.db.session import make_engine, make_session_factory
from app.pipeline.run_config import resolve_run_config
from app.storage.base import StorageBackend
from app.storage.factory import get_storage
from app.storage.keys import new_audio_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("import_librispeech_gpu")

DATASET_NAME = "librispeech-test-clean"
REFERENCE_SOURCE = "librispeech-test-clean"
REFERENCE_LANGUAGE = "en"


def parse_trans_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            result[parts[0]] = parts[1]
    return result


def flac_to_wav_bytes(flac_path: Path) -> tuple[bytes, int, float]:
    with sf.SoundFile(str(flac_path)) as src:
        sample_rate = src.samplerate
        data = src.read(dtype="float32", always_2d=False)
        duration = len(data) / sample_rate
    buf = io.BytesIO()
    with sf.SoundFile(buf, mode="w", samplerate=sample_rate,
                      channels=1 if data.ndim == 1 else data.shape[1],
                      subtype="PCM_16", format="WAV") as out:
        out.write(data)
    buf.seek(0)
    return buf.read(), sample_rate, duration


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def find_flac_files(source: Path) -> list[tuple[Path, str]]:
    pairs: list[tuple[Path, str]] = []
    for trans_file in sorted(source.rglob("*.trans.txt")):
        transcripts = parse_trans_file(trans_file)
        folder = trans_file.parent
        for utt_id in transcripts:
            flac_path = folder / f"{utt_id}.flac"
            if flac_path.exists():
                pairs.append((flac_path, utt_id))
            else:
                log.warning("missing FLAC for utterance %s", utt_id)
    return pairs


def _get_or_create_dataset(s, user_id: str, name: str) -> Dataset:
    ds = s.scalar(select(Dataset).where(Dataset.user_id == user_id, Dataset.name == name))
    if ds is None:
        ds = Dataset(user_id=user_id, name=name)
        s.add(ds)
        s.flush()
    return ds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/test-clean/LibriSpeech/test-clean")
    parser.add_argument("--model", default="faster-whisper-tiny")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        log.error("Source path does not exist: %s", source)
        sys.exit(1)

    # Use GPU settings → Postgres
    settings = get_gpu_settings()
    engine = make_engine(settings.database_url)
    session_factory = make_session_factory(engine)
    storage: StorageBackend = get_storage(settings)

    pairs = find_flac_files(source)
    log.info("Found %d FLAC files under %s", len(pairs), source)
    if args.limit:
        pairs = pairs[:args.limit]
        log.info("Limited to %d files", args.limit)

    if args.dry_run:
        for p, u in pairs[:5]:
            log.info("[dry-run] %s", p)
        return

    # Get user from Postgres
    from app.db.models import User
    with session_factory() as s:
        user = s.scalar(select(User))
        if user is None:
            log.error("No users in Postgres. Run: python scripts/register_keys_pg.py")
            sys.exit(1)
        user_id = user.id
        log.info("Importing as user: %s (%s)", user.username, user_id)

    # Parse all transcripts
    all_transcripts: dict[str, str] = {}
    for trans_file in source.rglob("*.trans.txt"):
        all_transcripts.update(parse_trans_file(trans_file))

    # Resolve run config using GPU registry
    registry = ModelRegistryGPU(settings)
    if args.model not in registry.names():
        log.error("Unknown model '%s'. Available: %s", args.model, registry.names())
        sys.exit(1)
    run_config_resolved, run_config_hash = resolve_run_config(
        args.model, {}, registry.options_model(args.model)
    )

    # Import GPU celery task for enqueueing
    from gpu.celery_app_gpu import process_run_gpu

    skipped = imported = failed = 0
    t0 = time.perf_counter()

    for i, (flac_path, utt_id) in enumerate(pairs):
        ref_text = all_transcripts.get(utt_id)
        if ref_text is None:
            log.warning("No transcript for %s, skipping", utt_id)
            skipped += 1
            continue

        try:
            wav_bytes, sample_rate, duration = flac_to_wav_bytes(flac_path)
        except Exception as e:
            log.error("Failed to read %s: %s", flac_path, e)
            failed += 1
            continue

        checksum = sha256_of_bytes(wav_bytes)

        with session_factory() as s:
            # Check if already imported
            existing = s.scalar(select(Audio).where(Audio.checksum == checksum))
            if existing is not None:
                ref = s.scalar(select(ReferenceTranscript).where(ReferenceTranscript.audio_id == existing.id))
                if ref is None:
                    s.add(ReferenceTranscript(audio_id=existing.id, text=ref_text,
                                              language=REFERENCE_LANGUAGE, source=REFERENCE_SOURCE))
                    s.commit()
                skipped += 1
                if (i + 1) % 100 == 0:
                    log.info("progress %d/%d  imported=%d skipped=%d failed=%d",
                             i + 1, len(pairs), imported, skipped, failed)
                continue

            key = new_audio_key(".wav")
            try:
                blob = storage.save_audio(key, io.BytesIO(wav_bytes))
            except Exception as e:
                log.error("Storage upload failed for %s: %s", utt_id, e)
                failed += 1
                continue

            dataset = _get_or_create_dataset(s, user_id, DATASET_NAME)
            audio = Audio(user_id=user_id, storage_key=key, checksum=blob.sha256,
                          duration=duration, original_sample_rate=sample_rate,
                          size_bytes=blob.size_bytes, dataset_id=dataset.id)
            s.add(audio)
            s.flush()
            s.add(ReferenceTranscript(audio_id=audio.id, text=ref_text,
                                      language=REFERENCE_LANGUAGE, source=REFERENCE_SOURCE))
            run = RunRepo(s).add(audio.id, args.model, run_config_resolved, config_hash=run_config_hash)
            s.commit()

            # Enqueue via GPU task
            process_run_gpu.delay(run.id)
            imported += 1

        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            remaining = (len(pairs) - i - 1) / rate if rate > 0 else 0
            log.info("progress %d/%d  imported=%d skipped=%d failed=%d  %.1f files/s  eta %.0fs",
                     i + 1, len(pairs), imported, skipped, failed, rate, remaining)

    elapsed = time.perf_counter() - t0
    log.info("done in %.1fs  imported=%d  skipped=%d  failed=%d", elapsed, imported, skipped, failed)
    log.info("Runs queued as 'asr.process_run_gpu'. Worker will pick them up now.")


if __name__ == "__main__":
    main()
