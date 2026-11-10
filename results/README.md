# Results

```
results/
├── CPU/       baselines and load tests from the MacBook (CPU)
├── GPU/       baselines and load tests from the RTX 3050 laptop
├── exports/   per-file exports (scripts/export_runs.py): how results move between the two databases
└── comparisons/  compare_runs.py --markdown output
```

## Baseline files

Written by `scripts/run_baseline.py`:

```
results/<CPU|GPU>/baseline_<model>_<device>-<compute_type>_<YYYYMMDDTHHMMSS>.json
```

| Field | Meaning |
|---|---|
| `model`, `dataset`, `timestamp`, `git_commit` | What ran, on which data, from which code |
| `config_hash`, `config` | The fully resolved run config and its hash (the leaderboard key) |
| `device`, `compute_type` | Hardware and precision the runs asked for (`config.model_options`) |
| `worker_concurrency` | Celery workers that processed the runs. Speed is comparable only at equal values; use 1 |
| `runtime` | Where it actually ran: device, precision, GPU name, library versions, platform |
| `normalization` | Text normalization of every score in the file (`whisper-en-v1`, or `basic-v1`) |
| `runs_total / runs_done / runs_failed / runs_unscored` | Counts |
| `wer.micro` | **Primary metric.** Total word errors / total reference words. Never average per-file WERs |
| `wer.per_file_mean`, `wer.p10 … p95` | Per-file WER distribution (for the tail) |
| `cer.micro` | Micro-averaged character error rate |
| `errors` | Total substitutions, deletions, insertions |
| `latency_s` | Per-file time from claim to completion: fetch, prepare (resample + VAD or cache), inference, scoring. Not queue wait |
| `rtf` | latency / audio duration per file; below 1 is faster than real time |
| `stages_s` | Mean, p50, p95 of `prepare_s`, `inference_s`, `score_s` |
| `inference_rtf` | Model time only / audio duration: the number to compare across hardware |
| `cache_hit_rate` | Share of runs whose chunk list / resampled audio came from the cache |
| `throughput` | Audio hours processed per wall-clock hour, first claim to last completion |
| `peak_rss_mb_max` | Largest peak resident memory of a worker process (not on Windows) |

`python scripts/report_baseline.py <files>` prints files and warns when normalization, dataset, worker count or
precision differ. For a WER difference with a confidence interval, export both runs and use
`python -m scripts.compare_runs A.jsonl.gz B.jsonl.gz`.

## Baselines recorded before Part H (read with care)

| File | What it is | Caveats |
|---|---|---|
| `CPU/baseline_faster-whisper-tiny_20261101T105820.json` (and the same run re-reported as `…20261102T083121.json`) | tiny, CPU int8, 8 Celery workers: WER 9.33%, RTF 1.58 | 8 workers shared one CPU (speed = contention); `basic-v1` scoring; temperature fallback active |
| `GPU/baseline_faster-whisper-tiny_20261108T234501.json` | tiny, RTX 3050 float16, 1 worker: WER 9.14%, RTF 0.149 | `basic-v1` scoring; temperature fallback active; no stage timings |

"Temperature fallback active": before Part H the configured `temperature` was never passed to faster-whisper, so
its fallback schedule (0.0 to 1.0) ran. New runs decode exactly as configured and get new config hashes.

No expected ranges are listed here on purpose: earlier guesses (e.g. 0.1 to 0.3x RTF for tiny on CPU) were wrong.
Only measured numbers belong in this folder.

## Load tests

`stage1_<N>users_*.csv/html` are Locust exports. Every load test so far used the **stub** model (Locust sent no
`model_version`), so they measure the API, queue and database, not inference. The GPU folder's load tests ran
about 90 s per stage on the Windows laptop, against about 5 minutes on the Mac, so the two sets are not
comparable stage by stage. Real-model load tests: see `locustfile.py` (`LOCUST_MODEL`, `LOCUST_AUDIO`).
