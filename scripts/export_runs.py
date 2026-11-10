"""Export one model + config's runs on a dataset to a portable file, keyed by utterance ID.

The CPU and GPU machines have separate databases whose audio ids differ, so results move between them as these
exports and are compared with scripts/compare_runs.py, matched on audio.external_id (the LibriSpeech utterance ID).

    python -m scripts.export_runs --model faster-whisper-tiny --machine mac-cpu            # the only config
    python -m scripts.export_runs --model faster-whisper-tiny --config-hash 3fa1 --machine rtx3050

Writes results/exports/<machine>_<model>_<hash8>_<normalization>.jsonl.gz (one JSON line per file) and a
.meta.json beside it (config, runtime, summary, git commit). Exports are small (~1-2 MB) and are committed.
Export BEFORE re-scoring with another normalization: re-scoring overwrites the stored scores.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select

from app.config import get_settings
from app.db.models import Audio, Dataset, ReferenceTranscript, RunStatus, TranscriptionRun
from app.db.session import make_engine, make_session_factory
from app.evaluation.summary import git_commit, summarize

EXPORT_FORMAT = 1
_STATUS_RANK = {RunStatus.done: 2, RunStatus.failed: 1}


def config_hashes(s, dataset_id: str, model: str) -> dict[str | None, int]:
    rows = s.execute(select(TranscriptionRun.config_hash, TranscriptionRun.id)
                     .join(Audio, Audio.id == TranscriptionRun.audio_id)
                     .where(Audio.dataset_id == dataset_id, TranscriptionRun.model_version == model)).all()
    out: dict[str | None, int] = {}
    for h, _ in rows:
        out[h] = out.get(h, 0) + 1
    return out


def collect(s, dataset_id: str, model: str, cfg_hash: str) -> tuple[list[dict], dict]:
    """One row per audio file (its latest done run, else its latest failed run) and the config."""
    q = (select(TranscriptionRun, Audio, ReferenceTranscript.text)
         .join(Audio, Audio.id == TranscriptionRun.audio_id)
         .outerjoin(ReferenceTranscript, ReferenceTranscript.audio_id == Audio.id)
         .where(Audio.dataset_id == dataset_id, TranscriptionRun.model_version == model,
                TranscriptionRun.config_hash == cfg_hash,
                TranscriptionRun.status.in_([RunStatus.done, RunStatus.failed])))
    best: dict[str, tuple] = {}
    for run, audio, ref in s.execute(q):
        key = (_STATUS_RANK[run.status], run.created_at)
        if audio.id not in best or key > best[audio.id][0]:
            best[audio.id] = (key, run, audio, ref)
    rows, config = [], None
    for _, run, audio, ref in best.values():
        config = run.config
        m = run.metrics or {}
        rows.append({
            "utt_id": audio.external_id, "audio_checksum": audio.checksum, "duration_s": audio.duration,
            "reference": ref, "transcript": run.transcript, "status": run.status.value, "error": run.error,
            "wer": run.wer, "cer": run.cer, "normalization": m.get("normalization"),
            **{k: m.get(k) for k in ("ref_words", "word_substitutions", "word_deletions", "word_insertions",
                                     "ref_chars", "char_substitutions", "char_deletions", "char_insertions")},
            "latency_s": run.latency, "profile": run.profile, "started_at": run.started_at.isoformat()
            if run.started_at else None, "run_id": run.id, "config_hash": run.config_hash,
        })
    rows.sort(key=lambda r: r["utt_id"] or "")
    return rows, config


def summary_rows(rows: list[dict]) -> list[dict]:
    return [{"status": r["status"], "wer": r["wer"], "latency": r["latency_s"], "profile": r["profile"],
             "started_at": r["started_at"], "duration": r["duration_s"],
             "metrics": {k: r[k] for k in ("ref_words", "word_substitutions", "word_deletions", "word_insertions",
                                           "ref_chars", "char_substitutions", "char_deletions", "char_insertions",
                                           "normalization")} if r["wer"] is not None else None}
            for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a model + config's runs for cross-machine comparison")
    ap.add_argument("--model", required=True)
    ap.add_argument("--machine", required=True, help="label for where the runs ran, e.g. mac-cpu or rtx3050")
    ap.add_argument("--dataset", default="librispeech-test-clean")
    ap.add_argument("--config-hash", help="full hash or a unique prefix; optional when only one exists")
    ap.add_argument("--worker-concurrency", type=int, help="Celery workers that ran these runs (for speed comparisons)")
    ap.add_argument("--out", help="default: results/exports/<machine>_<model>_<hash8>_<normalization>.jsonl.gz")
    a = ap.parse_args()

    st = get_settings()
    sf = make_session_factory(make_engine(st.database_url))
    with sf() as s:
        ds = s.scalar(select(Dataset).where(Dataset.name == a.dataset))
        if ds is None:
            sys.exit(f"dataset {a.dataset!r} not found")
        hashes = config_hashes(s, ds.id, a.model)
        matching = [h for h in hashes if h and (a.config_hash is None or h.startswith(a.config_hash))]
        if len(matching) != 1:
            listing = "\n".join(f"  {h}  ({n} runs)" for h, n in hashes.items())
            sys.exit(f"{'no' if not matching else 'several'} config hashes match for {a.model} on {a.dataset}; "
                     f"pass --config-hash. Available:\n{listing}")
        cfg_hash = matching[0]
        rows, config = collect(s, ds.id, a.model, cfg_hash)

    missing = sum(r["utt_id"] is None for r in rows)
    if missing:
        sys.exit(f"{missing} files have no external_id (utterance ID). Run alembic upgrade head, then re-run "
                 "scripts/import_librispeech.py on the same split to backfill, and export again.")
    norms = {r["normalization"] for r in rows if r["wer"] is not None}
    if len(norms) > 1:
        sys.exit(f"runs are scored under several normalizations {sorted(map(str, norms))}; re-score first")
    normalization = next(iter(norms), None) or "unscored"

    out = Path(a.out or f"results/exports/{a.machine}_{a.model}_{cfg_hash[:8]}_{normalization}.jsonl.gz")
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, default=str) + "\n")
    meta = {"format": EXPORT_FORMAT, "model": a.model, "dataset": a.dataset, "machine": a.machine,
            "config_hash": cfg_hash, "config": config, "normalization": normalization,
            "worker_concurrency": a.worker_concurrency,
            "exported_at": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(),
            "files": len(rows), "summary": summarize(summary_rows(rows))}
    meta_path = out.with_name(out.name.replace(".jsonl.gz", "") + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    w = meta["summary"].get("wer", {}).get("micro")
    print(f"exported {len(rows)} files ({meta['summary']['runs_done']} done, WER {w}) to {out}\nmeta: {meta_path}")


if __name__ == "__main__":
    main()
