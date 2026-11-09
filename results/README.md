# Results

## Folder layout

```
results/
├── CPU/   ← all results from CPU runs (existing baseline + load tests)
├── GPU/   ← all results from GPU runs (populated after GPU experiments)
└── README.md  ← this file
```

Save new CPU results to `results/CPU/` and new GPU results to `results/GPU/`
so the two sets stay separate and are easy to compare.

---

# Baseline Results

Each file is a snapshot of one full evaluation run over a dataset.

## File naming

```
baseline_<model-version>_<YYYYMMDDTHHMMSS>.json
```

## Fields

| Field | Description |
|---|---|
| `model` | Model version string (matches `model_version` in the DB) |
| `dataset` | Dataset name (matches `datasets.name` in the DB) |
| `timestamp` | UTC ISO-8601 timestamp when the report was generated |
| `runs_total` | Total run rows included in this report |
| `runs_done` | Runs that finished successfully |
| `runs_failed` | Runs that errored |
| `runs_unscored` | Done runs with no WER (missing reference transcript) |
| `wer.micro` | **Primary metric.** Micro-averaged WER: total word errors / total reference words across the whole corpus. This is the correct way to aggregate WER — do not average per-file WERs. |
| `wer.per_file_mean` | Mean of per-file WERs (for reference only; biased by short files) |
| `wer.p50 / p90 / p95` | Percentiles of the per-file WER distribution |
| `cer.micro` | Micro-averaged character error rate |
| `latency_s.mean/p50/p95` | Wall-clock transcription time per file (seconds) |
| `rtf.mean/p50/p95` | Real-time factor = latency / audio_duration. RTF < 1.0 = faster than real-time |
| `total_ref_words` | Total reference words across all scored files |
| `total_word_errors` | Total substitutions + deletions + insertions |

## Comparing results

```bash
# print the latest baseline
python scripts/report_baseline.py

# compare two baselines
python scripts/report_baseline.py results/baseline_faster-whisper-tiny_*.json
```

## Expected ranges (CPU, Apple Silicon M-series)

| Model | WER (test-clean) | RTF (CPU) |
|---|---|---|
| faster-whisper-tiny | ~5–8% | ~0.1–0.3x |
| faster-whisper-base | ~4–6% | ~0.2–0.5x |
| faster-whisper-small | ~3–4% | ~0.5–1.0x |

RTF < 1.0 means faster than real-time. All numbers are approximate and
depend on audio length, VAD chunk count, and hardware.
