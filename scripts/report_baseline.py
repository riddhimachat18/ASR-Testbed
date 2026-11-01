"""Print a formatted comparison table from one or more baseline JSON files.

Usage
─────
    # show the latest baseline
    python scripts/report_baseline.py

    # show a specific file
    python scripts/report_baseline.py results/baseline_faster-whisper-tiny_20261101T120000.json

    # compare two baselines side-by-side
    python scripts/report_baseline.py results/baseline_faster-whisper-tiny_*.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_DIR = Path("results")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def fmt_pct(v) -> str:
    if v is None:
        return "  N/A "
    return f"{v * 100:5.2f}%"


def fmt_f(v, decimals: int = 3) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def print_report(data: dict) -> None:
    model = data.get("model", "?")
    ts = data.get("timestamp", "?")
    ds = data.get("dataset", "?")
    print(f"\n{'═' * 68}")
    print(f"  Model:    {model}")
    print(f"  Dataset:  {ds}")
    print(f"  Run at:   {ts}")
    print(f"{'─' * 68}")

    r = data.get("runs_done", 0)
    f_ = data.get("runs_failed", 0)
    u = data.get("runs_unscored", 0)
    print(f"  Runs:     {r} done  /  {f_} failed  /  {u} unscored")

    w = data.get("wer", {})
    c = data.get("cer", {})
    print(f"\n  WER (word error rate)")
    print(f"    micro-avg : {fmt_pct(w.get('micro'))}")
    print(f"    per-file  : mean={fmt_pct(w.get('per_file_mean'))}  "
          f"p50={fmt_pct(w.get('p50'))}  p90={fmt_pct(w.get('p90'))}  p95={fmt_pct(w.get('p95'))}")
    print(f"\n  CER (character error rate)")
    print(f"    micro-avg : {fmt_pct(c.get('micro'))}")

    lat = data.get("latency_s", {})
    rtf = data.get("rtf", {})
    print(f"\n  Latency")
    print(f"    mean={fmt_f(lat.get('mean'), 2)}s  p50={fmt_f(lat.get('p50'), 2)}s  "
          f"p95={fmt_f(lat.get('p95'), 2)}s")
    print(f"\n  Real-time factor  (< 1.0 = faster than real-time)")
    print(f"    mean={fmt_f(rtf.get('mean'))}  p50={fmt_f(rtf.get('p50'))}  "
          f"p95={fmt_f(rtf.get('p95'))}")

    tw = data.get("total_ref_words")
    te = data.get("total_word_errors")
    if tw:
        print(f"\n  Corpus:   {tw:,} reference words  /  {te:,} word errors")
    print(f"{'═' * 68}\n")


def main() -> None:
    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        paths = sorted(RESULTS_DIR.glob("baseline_*.json"))
        if not paths:
            print(f"No baseline files found in {RESULTS_DIR}/")
            print("Run:  python scripts/run_baseline.py --model faster-whisper-tiny")
            return
        paths = [paths[-1]]  # most recent

    for path in paths:
        if not path.exists():
            print(f"File not found: {path}")
            continue
        data = load(path)
        print_report(data)
        print(f"  Source: {path}")


if __name__ == "__main__":
    main()
