"""Aggregate a set of finished runs into one baseline summary (used by run_baseline.py and export_runs.py).

Rules, so that two summaries are comparable:
  * WER and CER are micro-averaged: total errors / total reference words (characters).
  * A summary covers ONE normalization; mixed normalizations are refused (rescore first).
  * Percentiles use the same index rule as the pre-Part H baseline files: sorted[min(int(p * n), n - 1)].
  * RTF = latency / audio duration per file (latency covers fetch, prepare, inference and scoring, not queue
    wait). inference_rtf uses only the model's own time (profile.timings.inference_s), when recorded.
  * Throughput = audio hours / wall-clock hours from the first claim to the last completion. It depends on the
    worker count, so worker_concurrency is recorded next to it.
"""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone

STAGES = ("prepare_s", "inference_s", "score_s")
_ERR_KEYS = ("word_substitutions", "word_deletions", "word_insertions")
_CHAR_ERR_KEYS = ("char_substitutions", "char_deletions", "char_insertions")


class MixedNormalization(ValueError):
    pass


def pct(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    return round(sorted_vals[min(int(p * len(sorted_vals)), len(sorted_vals) - 1)], 4)


def _dist(vals: list[float], digits: int = 4) -> dict:
    s = sorted(vals)
    return {"mean": round(sum(s) / len(s), digits) if s else None, "p50": pct(s, 0.50), "p95": pct(s, 0.95)}


def _ts(dt) -> float | None:
    if dt is None:
        return None
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    if dt.tzinfo is None:  # SQLite returns naive UTC datetimes
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=5, check=True).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def summarize(rows: list[dict]) -> dict:
    """rows: dicts with status ('done'/'failed'), wer, metrics, latency, profile, started_at, duration."""
    done = [r for r in rows if r["status"] == "done" and r.get("wer") is not None and r.get("metrics")]
    failed = [r for r in rows if r["status"] == "failed"]
    unscored = [r for r in rows if r["status"] == "done" and (r.get("wer") is None or not r.get("metrics"))]
    norms = {r["metrics"].get("normalization") for r in done}
    if len(norms) > 1:
        raise MixedNormalization(f"runs are scored under several normalizations {sorted(map(str, norms))}; "
                                 "rescore them with scripts/evaluate.py --normalization first")
    out: dict = {"runs_total": len(rows), "runs_done": len(done), "runs_failed": len(failed),
                 "runs_unscored": len(unscored), "normalization": next(iter(norms), None)}
    if not done:
        return out

    ref_words = sum(r["metrics"]["ref_words"] for r in done)
    errs = {k: sum(r["metrics"].get(k, 0) for r in done) for k in _ERR_KEYS}
    ref_chars = sum(r["metrics"].get("ref_chars", 0) for r in done)
    char_errs = sum(r["metrics"].get(k, 0) for r in done for k in _CHAR_ERR_KEYS)
    per_file = sorted(r["wer"] for r in done)
    out["wer"] = {"micro": round(sum(errs.values()) / ref_words, 4),
                  "per_file_mean": round(sum(per_file) / len(per_file), 4),
                  **{f"p{int(p * 100)}": pct(per_file, p) for p in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95)}}
    out["cer"] = {"micro": round(char_errs / ref_chars, 4) if ref_chars else None}
    out["errors"] = {"substitutions": errs["word_substitutions"], "deletions": errs["word_deletions"],
                     "insertions": errs["word_insertions"]}

    timed = [r for r in done if r.get("latency") is not None]
    out["latency_s"] = _dist([r["latency"] for r in timed], 3)
    out["rtf"] = {**_dist([r["latency"] / r["duration"] for r in timed if r.get("duration")]),
                  "note": "rtf < 1.0 means faster than real-time"}
    profiled = [r for r in done if (r.get("profile") or {}).get("timings")]
    if profiled:
        out["stages_s"] = {st: _dist([r["profile"]["timings"][st] for r in profiled
                                      if st in r["profile"]["timings"]]) for st in STAGES}
        out["inference_rtf"] = _dist([r["profile"]["timings"]["inference_s"] / r["duration"] for r in profiled
                                      if "inference_s" in r["profile"]["timings"] and r.get("duration")])
        out["cache_hit_rate"] = {k: round(sum(bool((r["profile"].get("cache_hits") or {}).get(k))
                                              for r in profiled) / len(profiled), 3)
                                 for k in ("chunks", "resampled")}
        out["runtime"] = {k: v for k, v in (profiled[0]["profile"].get("runtime") or {}).items()
                          if k != "peak_rss_mb"}
        peaks = [r["profile"]["runtime"].get("peak_rss_mb") for r in profiled
                 if (r["profile"].get("runtime") or {}).get("peak_rss_mb") is not None]
        out["peak_rss_mb_max"] = max(peaks) if peaks else None

    starts = [_ts(r.get("started_at")) for r in timed if r.get("started_at") is not None]
    ends = [_ts(r["started_at"]) + r["latency"] for r in timed if r.get("started_at") is not None]
    audio_s = sum(r["duration"] for r in done if r.get("duration"))
    if starts:
        wall = max(ends) - min(starts)
        out["throughput"] = {"audio_hours": round(audio_s / 3600, 3), "wall_hours": round(wall / 3600, 3),
                             "audio_hours_per_wall_hour": round(audio_s / wall, 2) if wall > 0 else None}
    out["total_ref_words"] = ref_words
    out["total_word_errors"] = sum(errs.values())
    return out
