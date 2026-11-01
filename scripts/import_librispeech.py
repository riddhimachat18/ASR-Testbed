"""Import LibriSpeech test-clean into the ASR testbed.

Usage
─────
    # dry run — prints what would be imported, touches nothing
    python scripts/import_librispeech.py --source LibriSpeech/test-clean --dry-run

    # real import (idempotent: files already in storage are skipped via checksum)
    python scripts/import_librispeech.py --source LibriSpeech/test-clean

    # use a specific model so runs are queued immediately
    python scripts/import_librispeech.py --source LibriSpeech/test-clean --model faster-whisper-tiny

    # limit to N files (useful for smoke-testing)
    python scripts/import_librispeech.py --source LibriSpeech/test-clean --limit 50

What it does
────────────
1. Walks <source>/<speaker>/<chapter>/ looking for FLAC files and their
   matching .trans.txt transcripts.
2. For each FLAC, converts to WAV in memory (via soundfile / ffmpeg), uploads
   to the configured storage backend, and inserts Audio + ReferenceTranscript
   rows. All files are tagged with dataset_name="librispeech-test-clean".
3. If --model is given, a TranscriptionRun is created for each file so the
   Celery worker queue can pick them up immediately.
4. Files already present (matched by sha256 checksum) are skipped — so the
   script is safe to re-run after a crash.

Requirements
────────────
* The API server does NOT need to be running.
* ffmpeg must be on PATH (for FLAC → WAV conversion via soundfile).
* Run from the project root so that app/ is importable and .env is picked up.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import logging
import os
import sys
import time
from pathlib import Path

# ── make sure `app` is importable when run from the project root ───────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

import soundfile as sf
from sqlalchemy import select

from app.config import get_settings
from app.db.models import Audio, Dataset, ReferenceTranscript
from app.db.repositories import RunRepo, UserRepo
from app.db.session import make_engine, make_session_factory
from app.pipeline.registry import ModelRegistry
from app.pipeline.run_config import resolve_run_config
from app.storage.base import StorageBackend
from app.storage.factory import get_storage
from app.storage.keys import new_audio_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("import_librispeech")

DATASET_NAME = "librispeech-test-clean"
REFERENCE_SOURCE = "librispeech-test-clean"
REFERENCE_LANGUAGE = "en"


# ── transcript parsing ─────────────────────────────────────────────────────────

def parse_trans_file(path: Path) -> dict[str, str]:
    """Return {utterance_id: reference_text} from a .trans.txt file."""
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            result[parts[0]] = parts[1]
    return result


# ── audio conversion ───────────────────────────────────────────────────────────

def flac_to_wav_bytes(flac_path: Path) -> tuple[bytes, int, float]:
    """Read FLAC, return (wav_bytes, sample_rate, duration_s).
    Keeps original sample rate — resampling to 16 kHz is done by DerivedCache.
    """
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


# ── storage helpers ────────────────────────────────────────────────────────────

def _already_imported(s, checksum: str) -> Audio | None:
    return s.scalar(select(Audio).where(Audio.checksum == checksum))


def _get_or_create_dataset(s, user_id: str, name: str) -> Dataset:
    ds = s.scalar(select(Dataset).where(Dataset.user_id == user_id, Dataset.name == name))
    if ds is None:
        ds = Dataset(user_id=user_id, name=name)
        s.add(ds)
        s.flush()
    return ds


# ── main ───────────────────────────────────────────────────────────────────────

def find_flac_files(source: Path) -> list[tuple[Path, str]]:
    """Yield (flac_path, utterance_id) pairs for every FLAC under source."""
    pairs: list[tuple[Path, str]] = []
    for trans_file in sorted(source.rglob("*.trans.txt")):
        transcripts = parse_trans_file(trans_file)
        folder = trans_file.parent
        for utt_id, _text in transcripts.items():
            flac_path = folder / f"{utt_id}.flac"
            if flac_path.exists():
                pairs.append((flac_path, utt_id))
            else:
                log.warning("missing FLAC for utterance %s", utt_id)
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Import LibriSpeech test-clean")
    parser.add_argument("--source", default="LibriSpeech/test-clean",
                        help="Path to the test-clean directory (default: LibriSpeech/test-clean)")
    parser.add_argument("--model", default=None,
                        help="If set, create a TranscriptionRun for each file with this model version")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be imported without touching anything")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after importing this many files")
    parser.add_argument("--user", default=None,
                        help="Username to import under (default: first user in DB)")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        log.error("Source path does not exist: %s", source)
        sys.exit(1)

    settings = get_settings()
    engine = make_engine(settings.database_url)
    session_factory = make_session_factory(engine)
    storage: StorageBackend = get_storage(settings)

    pairs = find_flac_files(source)
    log.info("Found %d FLAC files under %s", len(pairs), source)
    if args.limit:
        pairs = pairs[: args.limit]
        log.info("Limited to %d files", len(pairs))

    if args.dry_run:
        for flac_path, utt_id in pairs[:10]:
            log.info("[dry-run] would import %s", flac_path)
        if len(pairs) > 10:
            log.info("[dry-run] ... and %d more", len(pairs) - 10)
        log.info("[dry-run] total: %d files", len(pairs))
        return

    # Need a user to own the audio
    with session_factory() as s:
        if args.user:
            from app.db.models import User
            user = s.scalar(select(User).where(User.username == args.user))
            if user is None:
                log.error("User '%s' not found", args.user)
                sys.exit(1)
        else:
            from app.db.models import User
            user = s.scalar(select(User))
            if user is None:
                log.error("No users in DB. Run: python scripts/create_user.py <username>")
                sys.exit(1)
        user_id = user.id
        log.info("Importing as user: %s (%s)", user.username, user_id)

    # Pre-parse all transcripts into a flat lookup
    all_transcripts: dict[str, str] = {}
    for trans_file in source.rglob("*.trans.txt"):
        all_transcripts.update(parse_trans_file(trans_file))

    # Resolve run config once if --model is set
    run_config_resolved = None
    run_config_hash = None
    if args.model:
        registry = ModelRegistry(settings)
        if args.model not in registry.names():
            log.error("Unknown model '%s'. Available: %s", args.model, registry.names())
            sys.exit(1)
        run_config_resolved, run_config_hash = resolve_run_config(
            args.model, {}, registry.options_model(args.model)
        )

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
            existing = _already_imported(s, checksum)
            if existing is not None:
                # Already in storage — still ensure reference and dataset are set
                ref = s.scalar(
                    select(ReferenceTranscript).where(ReferenceTranscript.audio_id == existing.id)
                )
                if ref is None:
                    s.add(ReferenceTranscript(
                        audio_id=existing.id, text=ref_text,
                        language=REFERENCE_LANGUAGE, source=REFERENCE_SOURCE,
                    ))
                    s.commit()
                skipped += 1
                if (i + 1) % 100 == 0:
                    log.info("progress %d/%d  imported=%d skipped=%d failed=%d",
                             i + 1, len(pairs), imported, skipped, failed)
                continue

            # Upload to storage
            key = new_audio_key(".wav")
            try:
                blob = storage.save_audio(key, io.BytesIO(wav_bytes))
            except Exception as e:
                log.error("Storage upload failed for %s: %s", utt_id, e)
                failed += 1
                continue

            dataset = _get_or_create_dataset(s, user_id, DATASET_NAME)

            audio = Audio(
                user_id=user_id,
                storage_key=key,
                checksum=blob.sha256,
                duration=duration,
                original_sample_rate=sample_rate,
                size_bytes=blob.size_bytes,
                dataset_id=dataset.id,
            )
            s.add(audio)
            s.flush()

            s.add(ReferenceTranscript(
                audio_id=audio.id,
                text=ref_text,
                language=REFERENCE_LANGUAGE,
                source=REFERENCE_SOURCE,
            ))

            if run_config_resolved is not None:
                run = RunRepo(s).add(
                    audio.id, args.model, run_config_resolved, config_hash=run_config_hash
                )
                _ = run  # will be picked up by the Celery worker

            s.commit()
            imported += 1

        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            remaining = (len(pairs) - i - 1) / rate if rate > 0 else 0
            log.info(
                "progress %d/%d  imported=%d skipped=%d failed=%d  "
                "%.1f files/s  eta %.0fs",
                i + 1, len(pairs), imported, skipped, failed, rate, remaining,
            )

    elapsed = time.perf_counter() - t0
    log.info(
        "done in %.1fs  imported=%d  skipped=%d  failed=%d",
        elapsed, imported, skipped, failed,
    )
    if args.model:
        log.info("Runs queued for model '%s'. Start workers to process them.", args.model)


if __name__ == "__main__":
    main()
