from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Audio, Dataset, ReferenceTranscript, RunStatus, TranscriptionRun
from app.evaluation.metrics import DEFAULT_NORMALIZATION, compute_metrics


def score_text(s: Session, audio_id: str, transcript: str,
               normalization: str = DEFAULT_NORMALIZATION) -> dict | None:
    """Score a transcript against the audio's reference. None if the audio has no reference.
    Returns {"wer", "cer", "metrics"}; wer and cer are None when the reference is unusable (empty)."""
    ref = s.scalar(select(ReferenceTranscript).where(ReferenceTranscript.audio_id == audio_id))
    if ref is None:
        return None
    m = compute_metrics(ref.text, transcript, normalization)
    if m is None:
        return {"wer": None, "cer": None,
                "metrics": {"error": "reference is empty after normalization", "normalization": normalization}}
    return {"wer": m["wer"], "cer": m["cer"], "metrics": m}


def evaluate_run(s: Session, run_id: str, normalization: str = DEFAULT_NORMALIZATION) -> bool:
    """(Re)score one finished run, e.g. after a reference is added or changed, or with another normalization.
    True if scored. Overwrites wer, cer and metrics; `profile` (timings) is untouched. Does not commit."""
    run = s.execute(select(TranscriptionRun).where(TranscriptionRun.id == run_id)
                    .execution_options(populate_existing=True)).scalar_one_or_none()
    if run is None or run.status is not RunStatus.done or run.transcript is None:
        return False
    scores = score_text(s, run.audio_id, run.transcript, normalization)
    if scores is None:
        return False
    run.wer, run.cer, run.metrics = scores["wer"], scores["cer"], scores["metrics"]
    return True


def evaluate_audio_runs(s: Session, audio_id: str, normalization: str = DEFAULT_NORMALIZATION) -> int:
    """Re-score every finished run of one audio file (used when its reference is added or changed)."""
    ids = s.scalars(select(TranscriptionRun.id).where(
        TranscriptionRun.audio_id == audio_id, TranscriptionRun.status == RunStatus.done)).all()
    return sum(evaluate_run(s, rid, normalization) for rid in ids)


def leaderboard(s: Session, user_id: str, dataset_name: str) -> list[dict] | None:
    """One row per (model_version, config_hash, normalization), micro-averaged over every scored run in the
    dataset. None if no such dataset. Normalization is part of the key: WERs under different normalizations
    are not comparable, so they are never added together."""
    ds = s.scalar(select(Dataset).where(Dataset.user_id == user_id, Dataset.name == dataset_name))
    if ds is None:
        return None
    rows = s.execute(
        select(TranscriptionRun.model_version, TranscriptionRun.config_hash, TranscriptionRun.config,
               TranscriptionRun.audio_id, TranscriptionRun.metrics)
        .join(Audio, Audio.id == TranscriptionRun.audio_id)
        .where(Audio.dataset_id == ds.id, TranscriptionRun.status == RunStatus.done,
               TranscriptionRun.wer.is_not(None))
    ).all()
    agg = defaultdict(lambda: defaultdict(int))
    audios = defaultdict(set)
    configs = {}
    for model, cfg_hash, cfg, audio_id, m in rows:
        key = (model, cfg_hash, m.get("normalization"))  # different settings or normalization = different row
        configs.setdefault(key, cfg)
        a = agg[key]
        a["runs"] += 1
        for k in ("ref_words", "word_substitutions", "word_deletions", "word_insertions",
                  "ref_chars", "char_substitutions", "char_deletions", "char_insertions"):
            a[k] += m[k]
        audios[key].add(audio_id)
    out = []
    for key, a in agg.items():
        model, cfg_hash, normalization = key
        word_err = a["word_substitutions"] + a["word_deletions"] + a["word_insertions"]
        char_err = a["char_substitutions"] + a["char_deletions"] + a["char_insertions"]
        out.append({"model_version": model, "config_hash": cfg_hash, "normalization": normalization,
                    "config": configs[key],
                    "runs": a["runs"], "audio_files": len(audios[key]),
                    "ref_words": a["ref_words"], "wer": word_err / a["ref_words"],
                    "cer": char_err / a["ref_chars"]})
    return sorted(out, key=lambda r: r["wer"])
