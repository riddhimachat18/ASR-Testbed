from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Audio, Dataset, ReferenceTranscript, RunStatus, TranscriptionRun
from app.evaluation.metrics import NORMALIZATION_VERSION, compute_metrics


def evaluate_run(s: Session, run_id: str) -> bool:
    """Score one finished run against its audio's reference, if it has one. Returns True if scored.
    Does not commit; the caller does."""
    run = s.execute(select(TranscriptionRun).where(TranscriptionRun.id == run_id)
                    .execution_options(populate_existing=True)).scalar_one_or_none()
    if run is None or run.status is not RunStatus.done or run.transcript is None:
        return False
    ref = s.scalar(select(ReferenceTranscript).where(ReferenceTranscript.audio_id == run.audio_id))
    if ref is None:
        return False
    m = compute_metrics(ref.text, run.transcript)
    if m is None:
        run.wer = run.cer = None
        run.metrics = {"error": "reference is empty after normalization", "normalization": NORMALIZATION_VERSION}
    else:
        run.wer, run.cer, run.metrics = m["wer"], m["cer"], m
    return True


def evaluate_audio_runs(s: Session, audio_id: str) -> int:
    """Re-score every finished run of one audio file (used when its reference is added or changed)."""
    ids = s.scalars(select(TranscriptionRun.id).where(
        TranscriptionRun.audio_id == audio_id, TranscriptionRun.status == RunStatus.done)).all()
    return sum(evaluate_run(s, rid) for rid in ids)


def leaderboard(s: Session, user_id: str, dataset_name: str) -> list[dict] | None:
    """Per model_version, micro-averaged over every scored run in the dataset. None if no such dataset."""
    ds = s.scalar(select(Dataset).where(Dataset.user_id == user_id, Dataset.name == dataset_name))
    if ds is None:
        return None
    rows = s.execute(
        select(TranscriptionRun.model_version, TranscriptionRun.audio_id, TranscriptionRun.metrics)
        .join(Audio, Audio.id == TranscriptionRun.audio_id)
        .where(Audio.dataset_id == ds.id, TranscriptionRun.status == RunStatus.done,
               TranscriptionRun.wer.is_not(None))
    ).all()
    agg = defaultdict(lambda: defaultdict(int))
    audios = defaultdict(set)
    for model, audio_id, m in rows:
        a = agg[model]
        a["runs"] += 1
        for k in ("ref_words", "word_substitutions", "word_deletions", "word_insertions",
                  "ref_chars", "char_substitutions", "char_deletions", "char_insertions"):
            a[k] += m[k]
        audios[model].add(audio_id)
    out = []
    for model, a in agg.items():
        word_err = a["word_substitutions"] + a["word_deletions"] + a["word_insertions"]
        char_err = a["char_substitutions"] + a["char_deletions"] + a["char_insertions"]
        out.append({"model_version": model, "runs": a["runs"], "audio_files": len(audios[model]),
                    "ref_words": a["ref_words"], "wer": word_err / a["ref_words"],
                    "cer": char_err / a["ref_chars"]})
    return sorted(out, key=lambda r: r["wer"])
