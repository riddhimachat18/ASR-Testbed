# Running the testbed on a GPU machine

Since Part H there is one code path for CPU and GPU (the old `gpu/` package is gone). A GPU machine differs
only in its settings: `DEVICE=cuda`, `COMPUTE_TYPE=float16`. Those defaults are written into every run's
`config.model_options`, so GPU runs get their own config hash and never mix with CPU runs on the leaderboard.

## Setup (Windows, RTX 3050 tested)

```powershell
# 1. CUDA torch first (pick the CUDA version that matches `nvidia-smi`), then the rest
pip uninstall torch torchaudio -y
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-gpu.txt

# 2. Settings: copy the example and select it
copy .env.gpu.example .env.gpu
$env:ASR_ENV_FILE=".env.gpu"          # bash: export ASR_ENV_FILE=.env.gpu

# 3. Infrastructure and schema
docker compose -f docker-compose-gpu-lite.yml up -d     # Postgres + Redis
alembic upgrade head

# 4. New API keys (the old ones leaked; see the design log, Part H.2) and utterance IDs
python -m scripts.rotate_keys
python scripts/import_librispeech.py --source LibriSpeech/test-clean   # imports, or backfills IDs

# 5. Worker and API (two terminals, both with ASR_ENV_FILE set)
celery -A app.celery_app worker --pool=solo --concurrency=1 --loglevel=info   # --pool=solo: Windows
uvicorn app.main:factory --factory --port 8001
```

Check CUDA is really used: `python -c "import ctranslate2; print(ctranslate2.get_supported_compute_types('cuda'))"`,
and after a run, its `profile.runtime` shows `device: cuda` and the GPU name.

## Measurement rules

- **One worker** (`--concurrency=1`) for every speed number; pass `--worker-concurrency 1` to `run_baseline.py`.
- **VAD on CPU** (`VAD_DEVICE=cpu`) on both machines, so chunk boundaries are identical and only the model differs.
- **Compare through exports:** `scripts/export_runs.py` on each machine, `scripts/compare_runs.py` anywhere.
- **Record the hardware once:** `nvidia-smi > results/GPU/hardware.txt`, plus driver and CUDA versions.

## Run matrix (hand-off, Part H)

```bash
python scripts/run_baseline.py --model faster-whisper-tiny --limit 20 --worker-concurrency 1   # smoke test first
python scripts/run_baseline.py --model faster-whisper-tiny --compute-type float16      --worker-concurrency 1
python scripts/run_baseline.py --model faster-whisper-tiny --compute-type int8_float16 --worker-concurrency 1
python scripts/run_baseline.py --model faster-whisper-tiny --compute-type int8         --worker-concurrency 1
python scripts/run_baseline.py --model faster-whisper-tiny --compute-type float32      --worker-concurrency 1
python scripts/run_baseline.py --model faster-whisper-base  --worker-concurrency 1
python scripts/run_baseline.py --model faster-whisper-small --worker-concurrency 1
python scripts/run_baseline.py --model faster-whisper-medium --worker-concurrency 1    # if VRAM allows
python scripts/run_baseline.py --model <best> --beam-size 1 --worker-concurrency 1
# decoding choice (Part H.7): fallback off + loop guard no_repeat_ngram_size=3 is the default; one sensitivity run with
# Whisper's random temperature fallback, and one repeat of it to measure its run-to-run variance
python scripts/run_baseline.py --model <best> --config '{"model_options":{"temperature_fallback":true}}' --worker-concurrency 1
python scripts/run_baseline.py --model <best> --config '{"model_options":{"temperature_fallback":true}}' --worker-concurrency 1
# then, per run:  python -m scripts.export_runs --model <model> --config-hash <prefix> --machine rtx3050 --worker-concurrency 1
```

## Troubleshooting

- `CUDA error: no kernel image is available for execution on the device`: the torch wheel does not match the
  driver. Check `nvidia-smi` and reinstall torch from the matching `cu1xx` index.
- `device "cuda" is not available` from CTranslate2: reinstall it (`pip install --force-reinstall ctranslate2`).
- A run fails with an error from `model.to("cuda")` in Silero: set `VAD_DEVICE=cpu` (the recommended setting).
  Before Part H this fell back to CPU silently; it now fails so a run never records a device it did not use.
- `compute_type 'float16' needs device 'cuda'`: `DEVICE` is not `cuda`; check `ASR_ENV_FILE` is set in this terminal.
