"""Dashboard data layer: turns runs, stage_events, audio_features and committed exports into plain rows.

Sources
  db      runs in this machine's database, with their stage events (finest detail, live)
  export  results/exports/*.jsonl.gz written by scripts/export_runs.py, e.g. the GPU teammate's runs. They have
          per-run timings (and, for runs traced since Part I, the per-stage breakdown) but no individual events.
          Audio features are looked up by utterance ID in this database, so GPU runs get them too.

Groups are sets of comparable runs: one model + config hash + normalization (+ machine for exports). An export
of runs that are also in this database is marked `duplicate` and hidden by default.

Everything here is read-only and returns JSON-ready dicts; app/dashboard/routes.py only wires HTTP.
"""
from __future__ import annotations

import gzip
import json
import math
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import func, select

from app.db.models import (Audio, AudioFeatures, Dataset, ReferenceTranscript, RunStatus, StageEvent,
                           TranscriptionRun as Run)
from app.observability.tracing import self_times, stage_category

FEATURE_COLS = ("duration_s", "rms_dbfs", "peak_dbfs", "clip_ratio", "speech_ratio", "n_speech_segments", "snr_db",
                "spectral_centroid_hz", "n_chunks", "padding_ratio")
MAX_GROUPS = 8


def _utc(ts: datetime | None) -> datetime | None:
    if ts is None:
        return None
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts


def _iso(ts: datetime | None) -> str | None:
    ts = _utc(ts)
    return ts.isoformat() if ts else None


def percentile(values: list[float], q: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = (len(vals) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def _r(x, n=4):
    return round(x, n) if isinstance(x, float) else x


# ── per-run rows ──────────────────────────────────────────────────────────────
def _device_of(profile: dict | None, config: dict | None, machine: str | None = None) -> tuple[str, str]:
    rt = (profile or {}).get("runtime") or {}
    opts = (config or {}).get("model_options") or {}
    device = rt.get("device") or opts.get("device")
    compute = rt.get("compute_type") or opts.get("compute_type")
    if device is None and machine:
        device = "cpu" if "cpu" in machine.lower() else "cuda"
    return device or "cpu", compute or ("int8" if (device or "cpu") == "cpu" else "?")


def _row_from_profile(row: dict, profile: dict | None) -> None:
    p = profile or {}
    t = p.get("timings") or {}
    dec = p.get("decoder") or {}
    row.update(prepare_s=t.get("prepare_s"), inference_s=t.get("inference_s"), score_s=t.get("score_s"),
               total_s=t.get("total_s"), n_chunks=p.get("n_chunks"), segments=dec.get("segments"),
               fallback_segments=dec.get("fallback_segments"), output_words=dec.get("output_words"),
               longest_repeat=dec.get("longest_repeat"), stages=p.get("stages"),
               cache_hits=p.get("cache_hits"), peak_rss_mb=((p.get("runtime") or {}).get("peak_rss_mb")))


def aggregate_run_events(events: list[dict]) -> dict:
    """Per-run figures from its stage events (latest attempt only)."""
    if not events:
        return {}
    last = max(e["attempt"] for e in events)
    events = [e for e in events if e["attempt"] == last]
    by_seq = {e["seq"]: e for e in events}
    out: dict = {"encoder_ms": 0.0, "encoder_calls": 0, "lang_ms": 0.0, "lang_calls": 0, "decoder_ms": 0.0,
                 "decoder_calls": 0, "tokens": 0, "features_ms": 0.0, "audio_decode_ms": 0.0, "slice_ms": 0.0,
                 "vad_ms": 0.0, "resample_ms": 0.0, "queue_ms": None, "model_get_ms": 0.0, "model_cold": False}
    logprobs, crs, fallbacks = [], [], 0
    for e in events:
        parent = by_seq.get(e["parent_seq"]) if e["parent_seq"] is not None else None
        cat = stage_category(e["stage"], parent["stage"] if parent else None)
        a = e.get("attrs") or {}
        d = e["duration_ms"]
        if cat == "model.encode":
            out["encoder_ms"] += d
            out["encoder_calls"] += 1
        elif cat == "model.language_detect":
            out["lang_ms"] += d
            out["lang_calls"] += 1
        elif cat == "model.decode":
            out["decoder_ms"] += d
            out["decoder_calls"] += 1
            out["tokens"] += a.get("tokens") or 0
            if a.get("avg_logprob") is not None:
                logprobs.append(a["avg_logprob"])
            if a.get("compression_ratio") is not None:
                crs.append(a["compression_ratio"])
            if (a.get("temperature") or 0) > 0:
                fallbacks += 1
        elif cat == "model.features":
            out["features_ms"] += d
        elif cat == "model.audio_decode":
            out["audio_decode_ms"] += d
        elif cat == "chunk.slice":
            out["slice_ms"] += d
        elif cat == "vad.detect":
            out["vad_ms"] += d
        elif cat == "resample.build":
            out["resample_ms"] += d
        elif cat == "queue_wait":
            out["queue_ms"] = d
        elif cat == "model.get":
            out["model_get_ms"] += d
            out["model_cold"] = out["model_cold"] or bool(a.get("cold"))
    out["avg_logprob"] = sum(logprobs) / len(logprobs) if logprobs else None
    out["min_logprob"] = min(logprobs) if logprobs else None
    out["max_compression_ratio"] = max(crs) if crs else None
    out["decode_fallbacks"] = fallbacks
    out["tokens_per_s"] = out["tokens"] / (out["decoder_ms"] / 1000.0) if out["decoder_ms"] > 0 else None
    out["stages"] = {k: round(v / 1000.0, 6) for k, v in self_times(events).items()}
    return {k: _r(v) for k, v in out.items()}


def _features_by(s, key: str) -> dict:
    """Audio features keyed by audio id or by utterance id (external_id)."""
    q = select(Audio.id, Audio.external_id, *[getattr(AudioFeatures, c) for c in FEATURE_COLS]).join(
        AudioFeatures, AudioFeatures.audio_id == Audio.id)
    out = {}
    for r in s.execute(q):
        k = r[0] if key == "audio_id" else r[1]
        if k is not None:
            out[k] = {c: r[i + 2] for i, c in enumerate(FEATURE_COLS)}
    return out


# ── groups ────────────────────────────────────────────────────────────────────
class DashboardData:
    """Caches the (cheap but not free) group scan for a few seconds; exports are cached until their file changes."""

    def __init__(self, session_factory, exports_dir: str | None, ttl_s: float = 15.0):
        self.sf, self.exports_dir, self.ttl = session_factory, exports_dir, ttl_s
        self._lock = threading.Lock()
        self._db_cache: tuple[float, list[dict]] | None = None
        self._exports: dict[str, tuple[float, dict, list[dict]]] = {}

    # db --------------------------------------------------------------------
    def _db_groups(self) -> list[dict]:
        """One aggregate per (model, config hash, normalization, dataset). JSON fields are extracted in SQL, so a
        scan of tens of thousands of runs stays cheap: the dashboard must not slow down the worker it measures
        when both run on one laptop."""
        with self._lock:
            if self._db_cache and time.monotonic() - self._db_cache[0] < self.ttl:
                return self._db_cache[1]
        m, cfg, prof = Run.metrics, Run.config, Run.profile
        q = (select(Run.model_version, Run.config_hash, Run.status, Run.latency, Audio.duration, Dataset.name,
                    Run.created_at, m["normalization"].as_string(), m["ref_words"].as_integer(),
                    m["word_substitutions"].as_integer(), m["word_deletions"].as_integer(),
                    m["word_insertions"].as_integer(), prof[("timings", "inference_s")].as_float(),
                    cfg[("model_options", "device")].as_string(), cfg[("model_options", "compute_type")].as_string(),
                    prof[("runtime", "machine")].as_string(), cfg[("decoding", "language")].as_string())
             .join(Audio, Audio.id == Run.audio_id).outerjoin(Dataset, Dataset.id == Audio.dataset_id))
        agg: dict[tuple, dict] = {}
        with self.sf() as s:
            traced_groups = set(s.execute(
                select(Run.model_version, Run.config_hash).join(StageEvent, StageEvent.run_id == Run.id)
                .where(StageEvent.stage == "run").distinct()).all())
            for (model, h, status, latency, dur, dataset, created, norm, ref_words, subs, dels, ins, inf,
                 device, compute, machine, language) in s.execute(q):
                key = (model, h, norm, dataset)
                g = agg.get(key)
                if g is None:
                    device = device or "cpu"
                    g = agg[key] = {"src": "db", "model": model, "config_hash": h, "normalization": norm,
                                    "dataset": dataset, "device": device,
                                    "compute_type": compute or ("int8" if device == "cpu" else "?"),
                                    "machine": machine, "language": language,
                                    "n": 0, "done": 0, "failed": 0, "queued": 0, "processing": 0,
                                    "errors": 0, "ref_words": 0, "latencies": [], "audio_s": 0.0,
                                    "infer_s": 0.0, "last": None, "traced": (model, h) in traced_groups}
                g["n"] += 1
                g[status.value] += 1
                created = _utc(created)
                if created and (g["last"] is None or created > g["last"]):
                    g["last"] = created
                if status is RunStatus.done:
                    if ref_words is not None:
                        g["errors"] += (subs or 0) + (dels or 0) + (ins or 0)
                        g["ref_words"] += ref_words
                    if latency is not None:
                        g["latencies"].append(latency)
                    if inf is not None and dur:
                        g["infer_s"] += inf
                        g["audio_s"] += dur
            s.rollback()
        groups = [self._finish_group(g, f"db|{g['model']}|{g['config_hash']}|{g['normalization']}|{g['dataset']}")
                  for g in agg.values()]
        with self._lock:
            self._db_cache = (time.monotonic(), groups)
        return groups

    @staticmethod
    def _finish_group(g: dict, key: str) -> dict:
        lat = g.pop("latencies")
        audio_s, infer_s = g.pop("audio_s"), g.pop("infer_s")
        errors, ref_words = g.pop("errors"), g.pop("ref_words")
        last = g.pop("last", None)
        h = g.get("config_hash") or "legacy"
        machine = g.get("machine_label") or ("this machine" if g["src"] == "db" else g.get("machine"))
        g.update(key=key, wer=_r(errors / ref_words) if ref_words else None, ref_words=ref_words,
                 latency_p50=_r(percentile(lat, 0.5)), latency_p95=_r(percentile(lat, 0.95)),
                 latency_mean=_r(sum(lat) / len(lat)) if lat else None,
                 rtf=_r(infer_s / audio_s) if audio_s else None, last=_iso(last) if isinstance(last, datetime) else last,
                 label=f"{(g['model'] or '?').replace('faster-whisper-', '')} · {g['device']} {g['compute_type']} · "
                       f"{h[:8]} · {machine}"
                       + (f" · {g['dataset'].replace('librispeech-', '')}" if g.get("dataset") else ""))
        return g

    # exports ---------------------------------------------------------------
    def _load_export(self, path: Path) -> tuple[dict, list[dict]]:
        mtime = path.stat().st_mtime
        cached = self._exports.get(str(path))
        if cached and cached[0] == mtime:
            return cached[1], cached[2]
        meta_path = path.with_name(path.name.replace(".jsonl.gz", ".meta.json"))
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        with gzip.open(path, "rt") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        self._exports[str(path)] = (mtime, meta, rows)
        return meta, rows

    def _export_groups(self, db_groups: list[dict]) -> list[dict]:
        if not self.exports_dir or not Path(self.exports_dir).is_dir():
            return []
        in_db = {(g["config_hash"], g["normalization"]) for g in db_groups}
        out = []
        for path in sorted(Path(self.exports_dir).glob("*.jsonl.gz")):
            try:
                meta, rows = self._load_export(path)
            except Exception:  # noqa: BLE001  a broken file must not break the dashboard
                continue
            first_profile = next((r.get("profile") for r in rows if r.get("profile")), None)
            device, compute = _device_of(first_profile, meta.get("config"), meta.get("machine"))
            g = {"src": "export", "file": path.name, "model": meta.get("model"),
                 "config_hash": meta.get("config_hash"), "normalization": meta.get("normalization"),
                 "dataset": meta.get("dataset"), "device": device, "compute_type": compute,
                 "machine": meta.get("machine"), "n": len(rows), "queued": 0, "processing": 0,
                 "done": sum(r.get("status") == "done" for r in rows),
                 "failed": sum(r.get("status") == "failed" for r in rows),
                 "errors": sum((r.get("word_substitutions") or 0) + (r.get("word_deletions") or 0)
                               + (r.get("word_insertions") or 0) for r in rows if r.get("ref_words")),
                 "ref_words": sum(r.get("ref_words") or 0 for r in rows),
                 "latencies": [r["latency_s"] for r in rows if r.get("latency_s") is not None],
                 "audio_s": sum(r.get("duration_s") or 0 for r in rows if (r.get("profile") or {}).get("timings")),
                 "infer_s": sum(((r.get("profile") or {}).get("timings") or {}).get("inference_s") or 0 for r in rows),
                 "traced": any((r.get("profile") or {}).get("stages") for r in rows),
                 "duplicate": (meta.get("config_hash"), meta.get("normalization")) in in_db,
                 "worker_concurrency": meta.get("worker_concurrency"),
                 "decoding": (meta.get("config") or {}).get("decoding"),
                 "model_options": (meta.get("config") or {}).get("model_options"),
                 "last": meta.get("exported_at")}
            out.append(self._finish_group(g, f"export|{path.name}"))
        return out

    def groups(self) -> list[dict]:
        db = self._db_groups()
        return sorted(db + self._export_groups(db),
                      key=lambda g: (g["src"] != "db", not g.get("traced"), g["model"] or "", g["device"],
                                     g["config_hash"] or "", g.get("dataset") or ""))

    # rows ------------------------------------------------------------------
    def rows(self, group_key: str) -> list[dict]:
        src, rest = group_key.split("|", 1)
        if src == "db":
            model, h, norm, dataset = rest.split("|")
            return self._db_rows(model, None if h == "None" else h, None if norm == "None" else norm,
                                 None if dataset == "None" else dataset, group_key)
        if src == "export":
            return self._export_rows(rest, group_key)
        raise KeyError(group_key)

    def _db_rows(self, model, h, norm, dataset, group_key) -> list[dict]:
        q = (select(Run.id, Run.status, Run.wer, Run.metrics, Run.latency, Run.profile, Run.config, Run.started_at,
                    Audio.id, Audio.external_id, Audio.duration)
             .join(Audio, Audio.id == Run.audio_id).outerjoin(Dataset, Dataset.id == Audio.dataset_id)
             .where(Run.model_version == model))
        q = q.where(Run.config_hash.is_(None) if h is None else Run.config_hash == h)
        q = q.where(Dataset.name.is_(None) if dataset is None else Dataset.name == dataset)
        out = []
        with self.sf() as s:
            feats = _features_by(s, "audio_id")
            items = [r for r in s.execute(q) if ((r[3] or {}).get("normalization") == norm)]
            ids = [r[0] for r in items]
            events: dict[str, list[dict]] = defaultdict(list)
            for i in range(0, len(ids), 900):
                for e in s.execute(select(StageEvent.run_id, StageEvent.attempt, StageEvent.seq, StageEvent.parent_seq,
                                          StageEvent.stage, StageEvent.duration_ms, StageEvent.attrs)
                                   .where(StageEvent.run_id.in_(ids[i:i + 900]))):
                    events[e[0]].append({"attempt": e[1], "seq": e[2], "parent_seq": e[3], "stage": e[4],
                                         "duration_ms": e[5], "attrs": e[6]})
            s.rollback()
        for rid, status, wer, metrics, latency, profile, config, started, audio_id, utt, dur in items:
            m = metrics or {}
            row = {"id": rid, "group": group_key, "utt": utt or audio_id[:8], "status": status.value, "wer": _r(wer),
                   "ref_words": m.get("ref_words"),
                   "errors": (m.get("word_substitutions", 0) + m.get("word_deletions", 0) + m.get("word_insertions", 0))
                   if m.get("ref_words") is not None else None,
                   "latency_s": _r(latency), "duration_s": dur, "started_at": _iso(started)}
            _row_from_profile(row, profile)
            ev = aggregate_run_events(events.get(rid, []))
            if ev:
                row.update(ev)
            row.update({f"f_{k}": v for k, v in (feats.get(audio_id) or {}).items()})
            out.append(row)
        return out

    def _export_rows(self, name: str, group_key: str) -> list[dict]:
        path = Path(self.exports_dir or "") / name
        if path.suffixes[-2:] != [".jsonl", ".gz"] or not path.exists():
            raise KeyError(name)
        _, rows = self._load_export(path)
        with self.sf() as s:
            feats = _features_by(s, "utt")
            s.rollback()
        out = []
        for r in rows:
            row = {"id": r.get("run_id"), "group": group_key, "utt": r.get("utt_id"), "status": r.get("status"),
                   "wer": _r(r.get("wer")), "ref_words": r.get("ref_words"),
                   "errors": ((r.get("word_substitutions") or 0) + (r.get("word_deletions") or 0)
                              + (r.get("word_insertions") or 0)) if r.get("ref_words") is not None else None,
                   "latency_s": _r(r.get("latency_s")), "duration_s": r.get("duration_s"),
                   "started_at": r.get("started_at")}
            _row_from_profile(row, r.get("profile"))
            row.update({f"f_{k}": v for k, v in (feats.get(r.get("utt_id")) or {}).items()})
            out.append(row)
        return out

    # stage breakdown --------------------------------------------------------
    def stages(self, group_key: str) -> dict:
        """Per stage category: share of total time (self time), and per-call duration percentiles where events
        exist. Falls back to the coarse profile timings (prepare / inference / score) for untraced runs."""
        rows = [r for r in self.rows(group_key) if r["status"] == "done"]
        traced = [r for r in rows if r.get("stages")]
        share: dict[str, float] = defaultdict(float)
        if traced:
            for r in traced:
                for k, v in r["stages"].items():
                    share[k] += v
            basis, n = "traced", len(traced)
        else:
            for r in rows:
                for k in ("prepare_s", "inference_s", "score_s"):
                    if r.get(k) is not None:
                        share[k.removesuffix("_s")] += r[k]
                if r.get("total_s") is not None:
                    share["other"] += max(r["total_s"] - sum(r.get(k) or 0 for k in ("prepare_s", "inference_s",
                                                                                        "score_s")), 0)
            basis, n = "profile", len(rows)
        total = sum(v for k, v in share.items() if k != "queue_wait") or 1.0  # share of processing time
        calls = self._call_stats(group_key) if group_key.startswith("db|") and traced else {}
        return {"basis": basis, "runs": n,
                "stages": sorted(({"stage": k, "total_s": _r(v),
                                   "share": None if k == "queue_wait" else _r(v / total), "per_run_ms":
                                   _r(1000.0 * v / n) if n else None, **calls.get(k, {})}
                                  for k, v in share.items()), key=lambda d: -d["total_s"])}

    def _call_stats(self, group_key: str) -> dict:
        _, model, h, norm, dataset = group_key.split("|")
        q = (select(StageEvent.stage, StageEvent.duration_ms, StageEvent.parent_seq, StageEvent.run_id,
                    StageEvent.seq, StageEvent.attempt)
             .join(Run, Run.id == StageEvent.run_id).join(Audio, Audio.id == Run.audio_id)
             .outerjoin(Dataset, Dataset.id == Audio.dataset_id).where(Run.model_version == model)
             .where(Run.config_hash.is_(None) if h == "None" else Run.config_hash == h)
             .where(Dataset.name.is_(None) if dataset == "None" else Dataset.name == dataset))
        durs: dict[str, list[float]] = defaultdict(list)
        with self.sf() as s:
            rows = list(s.execute(q))
            s.rollback()
        parent_stage = {(r[3], r[5], r[4]): r[0] for r in rows}
        for stage, d, parent_seq, run_id, seq, attempt in rows:
            cat = stage_category(stage, parent_stage.get((run_id, attempt, parent_seq)))
            durs[cat].append(d)
        return {k: {"calls": len(v), "p50_ms": _r(percentile(v, 0.5), 3), "p95_ms": _r(percentile(v, 0.95), 3),
                    "max_ms": _r(max(v), 3)} for k, v in durs.items()}

    # one run ---------------------------------------------------------------
    def run_detail(self, run_id: str) -> dict | None:
        with self.sf() as s:
            run = s.get(Run, run_id)
            if run is None:
                return None
            audio = s.get(Audio, run.audio_id)
            ref = s.scalar(select(ReferenceTranscript.text).where(ReferenceTranscript.audio_id == run.audio_id))
            feats = s.get(AudioFeatures, run.audio_id)
            evs = list(s.scalars(select(StageEvent).where(StageEvent.run_id == run_id)
                                 .order_by(StageEvent.attempt, StageEvent.seq)))
            last = max((e.attempt for e in evs), default=None)
            out = {"id": run.id, "status": run.status.value, "model": run.model_version, "config_hash": run.config_hash,
                   "config": run.config, "transcript": run.transcript, "reference": ref, "error": run.error,
                   "wer": run.wer, "metrics": run.metrics, "latency_s": run.latency, "profile": run.profile,
                   "utt": audio.external_id if audio else None, "duration_s": audio.duration if audio else None,
                   "features": {c: getattr(feats, c) for c in FEATURE_COLS} if feats else None,
                   "events": [{"seq": e.seq, "parent_seq": e.parent_seq, "stage": e.stage,
                               "started_at": _iso(e.started_at), "duration_ms": e.duration_ms,
                               "worker": e.worker, "attrs": e.attrs} for e in evs if e.attempt == last]}
            s.rollback()
        return out

    # live -----------------------------------------------------------------
    def live(self, since_id: int | None, window_s: float = 600.0, limit: int = 4000) -> dict:
        """New stage events since `since_id` (or the last `window_s` seconds on first call), the runs they
        finished, and current queue state."""
        now = datetime.now(timezone.utc)
        with self.sf() as s:
            counts = {st.value: 0 for st in RunStatus}
            for status, n in s.execute(select(Run.status, func.count()).group_by(Run.status)):
                counts[status.value] = n
            q = select(StageEvent.id, StageEvent.run_id, StageEvent.seq, StageEvent.parent_seq, StageEvent.stage,
                       StageEvent.started_at, StageEvent.duration_ms, StageEvent.worker, StageEvent.attrs)
            if since_id is None:
                q = q.where(StageEvent.started_at >= now - timedelta(seconds=window_s))
            else:
                q = q.where(StageEvent.id > since_id)
            evs = list(s.execute(q.order_by(StageEvent.id).limit(limit)))
            run_ids = list({e[1] for e in evs if e[4] == "run"})
            finished = {}
            for i in range(0, len(run_ids), 900):
                for rid, model, h, status, wer, latency, dur in s.execute(
                        select(Run.id, Run.model_version, Run.config_hash, Run.status, Run.wer, Run.latency,
                               Audio.duration).join(Audio, Audio.id == Run.audio_id)
                        .where(Run.id.in_(run_ids[i:i + 900]))):
                    finished[rid] = {"model": model, "config_hash": h, "status": status.value, "wer": wer,
                                     "latency_s": latency, "duration_s": dur}
            max_id = s.scalar(select(func.max(StageEvent.id)))
            s.rollback()
        last_id = evs[-1][0] if evs else (since_id if since_id is not None else (max_id or 0))
        keep = ("hit", "tokens", "cold", "index", "audio_s", "frames", "avg_logprob", "error", "tier",
                "compression_ratio", "temperature")
        return {"now": now.isoformat(), "last_id": last_id, "counts": counts,
                "events": [{"id": e[0], "run_id": e[1], "seq": e[2], "parent_seq": e[3], "stage": e[4],
                            "t": _iso(e[5]), "ms": round(e[6], 3), "worker": e[7],
                            "a": {k: v for k, v in (e[8] or {}).items() if k in keep}} for e in evs],
                "runs": finished}
