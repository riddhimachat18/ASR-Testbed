"""Compare two exports (scripts/export_runs.py) file by file, with a paired bootstrap confidence interval.

    python -m scripts.compare_runs results/exports/mac-cpu_….jsonl.gz results/exports/rtx3050_….jsonl.gz
    python -m scripts.compare_runs A.jsonl.gz B.jsonl.gz --markdown results/comparisons/tiny_int8_vs_fp16.md

Files are matched on utterance ID. The difference reported is B minus A (positive = B makes more errors).
Paired bootstrap: resample the matched files with replacement (the same files for A and B), recompute both
micro-WERs each time; the 2.5th and 97.5th percentiles of the difference give the 95% interval, and the
two-sided p is twice the share of resamples on the far side of zero. If the interval contains 0, the data do
not show a difference.

Refused: different normalizations. Warned: different datasets or reference texts, files present in only one
export, and speed differences when worker concurrency is unknown or differs.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ERR = ("word_substitutions", "word_deletions", "word_insertions")


def load(path: str | Path) -> tuple[dict, dict[str, dict]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    meta_path = path.with_name(path.name.replace(".jsonl.gz", "").replace(".jsonl", "") + ".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.setdefault("label", meta.get("machine") or path.stem)
    return meta, {r["utt_id"]: r for r in rows}


def _scored(r: dict | None) -> bool:
    return bool(r) and r.get("status") == "done" and r.get("wer") is not None and (r.get("ref_words") or 0) > 0


def paired_bootstrap(ref: np.ndarray, ea: np.ndarray, eb: np.ndarray, iters: int = 10_000,
                     seed: int = 0, chunk: int = 1_000) -> dict:
    """95% interval and two-sided p for micro-WER(B) - micro-WER(A) over paired files."""
    rng = np.random.default_rng(seed)
    n, diffs = len(ref), []
    for start in range(0, iters, chunk):
        idx = rng.integers(0, n, size=(min(chunk, iters - start), n))
        r = ref[idx].sum(axis=1)
        diffs.append(eb[idx].sum(axis=1) / r - ea[idx].sum(axis=1) / r)
    d = np.concatenate(diffs)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min(float((d <= 0).mean()), float((d >= 0).mean())))
    return {"ci95": (float(lo), float(hi)), "p": p, "iters": iters}


def compare(meta_a: dict, rows_a: dict, meta_b: dict, rows_b: dict, iters: int = 10_000) -> dict:
    na, nb = meta_a.get("normalization"), meta_b.get("normalization")
    norm_a = na or next((r["normalization"] for r in rows_a.values() if r.get("normalization")), None)
    norm_b = nb or next((r["normalization"] for r in rows_b.values() if r.get("normalization")), None)
    if norm_a != norm_b:
        raise ValueError(f"different normalizations ({norm_a} vs {norm_b}); re-score one side first")
    warnings = []
    if meta_a.get("dataset") != meta_b.get("dataset"):
        warnings.append(f"different datasets: {meta_a.get('dataset')} vs {meta_b.get('dataset')}")
    only_a = sorted(set(rows_a) - set(rows_b))
    only_b = sorted(set(rows_b) - set(rows_a))
    common = sorted(u for u in set(rows_a) & set(rows_b) if _scored(rows_a[u]) and _scored(rows_b[u]))
    unpaired = len(set(rows_a) & set(rows_b)) - len(common)
    if only_a or only_b:
        warnings.append(f"{len(only_a)} files only in A, {len(only_b)} only in B (left out)")
    if unpaired:
        warnings.append(f"{unpaired} shared files not scored on both sides (failed or unscored; left out)")
    ref_mismatch = [u for u in common if rows_a[u].get("reference") != rows_b[u].get("reference")]
    if ref_mismatch:
        warnings.append(f"{len(ref_mismatch)} files have different reference texts, e.g. {ref_mismatch[:3]}")
    if not common:
        raise ValueError("no files scored on both sides")

    ref = np.array([rows_a[u]["ref_words"] for u in common], dtype=np.int64)
    ea = np.array([sum(rows_a[u][k] for k in ERR) for u in common], dtype=np.int64)
    eb = np.array([sum(rows_b[u][k] for k in ERR) for u in common], dtype=np.int64)
    wa, wb = ea.sum() / ref.sum(), eb.sum() / ref.sum()
    boot = paired_bootstrap(ref, ea, eb, iters)
    per_file = sorted(((int(eb[i] - ea[i]), u) for i, u in enumerate(common)), reverse=True)
    res = {
        "a": meta_a.get("label"), "b": meta_b.get("label"), "normalization": norm_a, "files": len(common),
        "ref_words": int(ref.sum()), "wer_a": float(wa), "wer_b": float(wb), "diff": float(wb - wa),
        **boot, "errors_a": {k: int(sum(rows_a[u][k] for u in common)) for k in ERR},
        "errors_b": {k: int(sum(rows_b[u][k] for u in common)) for k in ERR},
        "files_b_worse": int((eb > ea).sum()), "files_b_better": int((eb < ea).sum()),
        "most_worse_in_b": [(u, d) for d, u in per_file[:10] if d > 0],
        "most_better_in_b": [(u, d) for d, u in sorted(per_file)[:10] if d < 0],
        "warnings": warnings,
    }
    res["speed"] = _speed(meta_a, rows_a, meta_b, rows_b, common, warnings)
    return res


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _speed(meta_a, rows_a, meta_b, rows_b, common, warnings) -> dict | None:
    ca, cb = meta_a.get("worker_concurrency"), meta_b.get("worker_concurrency")
    if ca is None or cb is None or ca != cb:
        warnings.append(f"worker concurrency {ca} vs {cb}: latency and RTF are not compared")
        return None

    def stats(rows):
        rtf = [r["latency_s"] / r["duration_s"] for r in (rows[u] for u in common)
               if r.get("latency_s") and r.get("duration_s")]
        inf = [r["profile"]["timings"]["inference_s"] / r["duration_s"] for r in (rows[u] for u in common)
               if (r.get("profile") or {}).get("timings", {}).get("inference_s") is not None and r.get("duration_s")]
        return {"rtf_mean": _mean(rtf), "inference_rtf_mean": _mean(inf)}
    return {"worker_concurrency": ca, "a": stats(rows_a), "b": stats(rows_b)}


def render(res: dict) -> str:
    lo, hi = res["ci95"]
    verdict = ("no difference shown (interval contains 0)" if lo <= 0 <= hi
               else ("B is worse" if res["diff"] > 0 else "B is better"))
    lines = [
        f"| | A: {res['a']} | B: {res['b']} |", "| --- | --- | --- |",
        f"| WER ({res['normalization']}, {res['files']} paired files) | {res['wer_a']:.2%} | {res['wer_b']:.2%} |",
        *(f"| {k.replace('word_', '').capitalize()} | {res['errors_a'][k]} | {res['errors_b'][k]} |" for k in ERR),
    ]
    if res.get("speed"):
        sa, sb = res["speed"]["a"], res["speed"]["b"]
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
        lines += [f"| Mean RTF ({res['speed']['worker_concurrency']} worker) | {fmt(sa['rtf_mean'])} | "
                  f"{fmt(sb['rtf_mean'])} |",
                  f"| Mean inference-only RTF | {fmt(sa['inference_rtf_mean'])} | {fmt(sb['inference_rtf_mean'])} |"]
    lines += ["", f"B - A = {res['diff'] * 100:+.2f} points; 95% paired-bootstrap interval "
              f"[{lo * 100:+.2f}, {hi * 100:+.2f}], p = {res['p']:.3f} ({res['iters']} resamples): {verdict}.",
              f"Files where B made more errors: {res['files_b_worse']}; fewer: {res['files_b_better']}."]
    if res["most_worse_in_b"]:
        lines.append("Largest increases in B: " + ", ".join(f"{u} (+{d})" for u, d in res["most_worse_in_b"]))
    if res["most_better_in_b"]:
        lines.append("Largest decreases in B: " + ", ".join(f"{u} ({d})" for u, d in res["most_better_in_b"]))
    lines += [f"Warning: {w}" for w in res["warnings"]]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired comparison of two run exports")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--iters", type=int, default=10_000)
    ap.add_argument("--markdown", help="also write the table and verdict to this file")
    a = ap.parse_args()
    try:
        res = compare(*load(a.a), *load(a.b), iters=a.iters)
    except ValueError as e:
        sys.exit(str(e))
    text = render(res)
    print(text)
    if a.markdown:
        Path(a.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(a.markdown).write_text(text + "\n")


if __name__ == "__main__":
    main()
