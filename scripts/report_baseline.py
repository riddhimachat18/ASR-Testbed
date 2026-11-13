"""Print one or more baseline JSON files, and warn when they are not comparable.

    python scripts/report_baseline.py                                   # latest file under results/
    python scripts/report_baseline.py results/GPU/baseline_*.json       # several, one after another

Two baselines are comparable on WER only under the same normalization and dataset, and on speed only with the
same worker concurrency (and, for hardware claims, the same precision). Files written before Part H lack
these fields; they are shown as "not recorded". For a WER difference with a confidence interval, export the
runs and use scripts/compare_runs.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_DIR = Path("results")
NOT_RECORDED = "not recorded"


def _pct(v) -> str:
    return "  N/A " if v is None else f"{v * 100:5.2f}%"


def _f(v, d: int = 3) -> str:
    return "N/A" if v is None else f"{v:.{d}f}"


def print_report(d: dict) -> None:
    print(f"\n{'═' * 72}")
    print(f"  Model:     {d.get('model', '?')}    dataset: {d.get('dataset', '?')}")
    print(f"  Run at:    {d.get('timestamp', '?')}    commit: {d.get('git_commit') or NOT_RECORDED}")
    print(f"  Hardware:  device={d.get('device') or NOT_RECORDED}  compute_type={d.get('compute_type') or NOT_RECORDED}"
          f"  workers={d.get('worker_concurrency') or NOT_RECORDED}")
    rt = d.get("runtime") or {}
    if rt:
        print(f"  Runtime:   {rt.get('gpu') or rt.get('platform', '')}  libs={rt.get('libs')}")
    print(f"  Config:    hash={(d.get('config_hash') or NOT_RECORDED)[:12]}  normalization="
          f"{d.get('normalization') or 'basic-v1 (pre-Part H)'}")
    print(f"{'─' * 72}")
    print(f"  Runs:      {d.get('runs_done', 0)} done / {d.get('runs_failed', 0)} failed / "
          f"{d.get('runs_unscored', 0)} unscored")
    w, c = d.get("wer", {}), d.get("cer", {})
    print(f"  WER:       micro {_pct(w.get('micro'))}   per-file mean {_pct(w.get('per_file_mean'))}  "
          f"p50 {_pct(w.get('p50'))}  p95 {_pct(w.get('p95'))}")
    print(f"  CER:       micro {_pct(c.get('micro'))}")
    if d.get("errors"):
        e = d["errors"]
        print(f"  Errors:    {e['substitutions']} sub / {e['deletions']} del / {e['insertions']} ins  "
              f"of {d.get('total_ref_words', 0):,} words")
    lat, rtf = d.get("latency_s", {}), d.get("rtf", {})
    print(f"  Latency:   mean {_f(lat.get('mean'), 2)} s  p50 {_f(lat.get('p50'), 2)} s  p95 {_f(lat.get('p95'), 2)} s")
    print(f"  RTF:       mean {_f(rtf.get('mean'))}  p50 {_f(rtf.get('p50'))}  p95 {_f(rtf.get('p95'))}"
          "   (< 1 = faster than real time)")
    if d.get("inference_rtf"):
        i = d["inference_rtf"]
        print(f"  Inference: RTF mean {_f(i.get('mean'))}  p95 {_f(i.get('p95'))}   (model time only)")
        st = d.get("stages_s", {})
        print("  Stages:    " + "  ".join(f"{k} {_f(v.get('mean'))} s" for k, v in st.items()))
        print(f"  Cache:     hit rate {d.get('cache_hit_rate')}")
        if d.get("self_time_s"):
            st = d["self_time_s"]
            print(f"  Self time per stage ({st['runs']} traced runs, mean per run, share of processing):")
            for k, v in st["stages"].items():
                share = "" if v["share"] is None else f"  {100 * v['share']:5.1f}%"
                print(f"    {k:<24} {v['mean']:8.3f} s{share}")
    if d.get("loops"):
        print(f"  Loops:     {d['loops']}")
    if d.get("throughput"):
        t = d["throughput"]
        print(f"  Throughput: {t['audio_hours']} h audio in {t['wall_hours']} h wall = "
              f"{t['audio_hours_per_wall_hour']} h/h")
    print(f"{'═' * 72}")


def comparability_warnings(docs: list[dict]) -> list[str]:
    warnings = []
    for field, why in (("normalization", "WER"), ("dataset", "WER"), ("worker_concurrency", "speed"),
                       ("compute_type", "speed and WER (precision)")):
        values = {str(doc.get(field)) for doc in docs}
        if len(values) > 1:
            warnings.append(f"{field} differs {sorted(values)}: {why} is not directly comparable")
    return warnings


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] or sorted(RESULTS_DIR.rglob("baseline_*.json"),
                                                       key=lambda p: p.stat().st_mtime)[-1:]
    if not paths:
        print(f"no baseline files under {RESULTS_DIR}/")
        return
    docs = []
    for p in paths:
        doc = json.loads(p.read_text())
        docs.append(doc)
        print_report(doc)
        print(f"  Source: {p}")
    for w in comparability_warnings(docs) if len(docs) > 1 else []:
        print(f"  WARNING: {w}")


if __name__ == "__main__":
    main()
