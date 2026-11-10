# GPU results (RTX 3050 laptop, Windows)

Same file format as `results/README.md`. Hardware details go in `hardware.txt` (`nvidia-smi` output, driver and
CUDA versions). How to run: `docs/GPU.md`.

| File | Run | Notes |
|---|---|---|
| `baseline_faster-whisper-tiny_20261108T234501.json` | tiny, float16, 1 worker: WER 9.14%, CER 3.89%, mean latency 1.04 s, RTF 0.149 | Before Part H: `basic-v1` scoring, temperature fallback active, no stage timings. The WER gap to the CPU baseline (9.33%) reflects float16 vs int8 and is untested for significance |
| `stage1_{20,100,500}users_*` | Locust, **stub model** | Measure the Windows API, Postgres and solo-pool setup, not the GPU; about 90 s per stage |

New baselines: `python scripts/run_baseline.py --model <model> --worker-concurrency 1` with `ASR_ENV_FILE=.env.gpu`.
Then export them (`python -m scripts.export_runs ... --machine rtx3050 --worker-concurrency 1`) and commit the export.
