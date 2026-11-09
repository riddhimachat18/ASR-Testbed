# GPU Pipeline — ASR Testbed

This directory is the GPU-mirrored version of the `app/` package.
Every file here has a counterpart in `app/`; the logic is identical unless
a comment says otherwise.

## What changed vs the CPU version

| File | What's different |
|---|---|
| `config_gpu.py` | Adds `DEVICE=cuda` and `COMPUTE_TYPE=float16` env vars |
| `pipeline/registry_gpu.py` | Passes `device` + `compute_type` from settings instead of hardcoded `"cpu"` / `"int8"` |
| `pipeline/faster_whisper_wrapper_gpu.py` | Defaults `device="cuda"`, `compute_type="float16"`; otherwise identical |
| `pipeline/silero_vad_gpu.py` | Moves model and wav tensor to `device` via `.to(device)`; otherwise identical |
| `pipeline/prepare_gpu.py` | Uses `SileroVadBackendGPU` instead of `SileroVadBackend` |
| `pipeline/worker_gpu.py` | Wired to GPU registry + preparer; otherwise identical |
| `pipeline/queues_gpu.py` | Routes Celery tasks to `process_run_gpu` task |
| `runtime_gpu.py` | Assembles GPU-specific components |
| `main_gpu.py` | FastAPI factory using GPU runtime |
| `celery_app_gpu.py` | Celery app with task name `asr.process_run_gpu` |

All CPU files (`app/`) are untouched.

## How to run (GPU server)

```bash
# 1. Install GPU torch (once — see requirements-gpu.txt for the exact command)
pip uninstall torch torchaudio -y
pip install torch==2.3.0+cu121 torchaudio==2.3.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. Install the rest of the requirements
pip install -r requirements-gpu.txt

# 3. Copy the GPU env file and edit if needed
cp .env.gpu .env.gpu.local   # optional: keep a local override

# 4. Run DB migrations (same DB as CPU — they share the schema)
alembic upgrade head

# 5. Start the GPU server  (use a different port to run alongside the CPU server)
uvicorn gpu.main_gpu:factory --factory --port 8001

# 6. Verify CUDA is active (check the startup logs for "loading faster-whisper model on GPU")
```

## How to run (GPU Celery worker)

```bash
# Terminal 1: GPU API
uvicorn gpu.main_gpu:factory --factory --port 8001

# Terminal 2: GPU Celery worker (set QUEUE_BACKEND=celery in .env.gpu)
celery -A gpu.celery_app_gpu worker --concurrency=1 --loglevel=info
```

`--concurrency=1` is recommended for GPU workers: one process holds the model
singleton in GPU memory.  Scale horizontally by running more workers.

## Comparing CPU vs GPU results

Both the CPU server and the GPU server write to the same database and results
directory.  After running both:

```bash
# CPU baseline (already done)
results/baseline_faster-whisper-tiny_20261101T105820.json

# GPU baseline (generated after your GPU run)
results/baseline_gpu_faster-whisper-tiny_<timestamp>.json

# Compare
python scripts/report_baseline.py \
    results/baseline_faster-whisper-tiny_*.json \
    results/baseline_gpu_faster-whisper-tiny_*.json
```

Key metrics to compare: **WER** (should be equal or better on GPU with float16)
and **RTF** (should be 10–50× lower on GPU).

## Environment variable reference

| Variable | CPU default | GPU default | Notes |
|---|---|---|---|
| `DEVICE` | *(not present)* | `cuda` | `"cpu"` / `"cuda"` / `"auto"` |
| `COMPUTE_TYPE` | *(not present)* | `float16` | `"float16"` / `"bfloat16"` / `"int8_float16"` |
| All others | same | same | Identical between CPU and GPU |

## Troubleshooting

**`RuntimeError: CUDA error: no kernel image is available for execution on the device`**
→ Your CUDA driver version doesn't match the torch wheel.
  Run `nvidia-smi` to check driver version, then reinstall torch with the matching `cu1xx` index URL.

**`ctranslate2.convert_model: device "cuda" is not available`**
→ CTranslate2 was installed without CUDA support.
  Reinstall: `pip install ctranslate2 --upgrade --force-reinstall`

**Silero-VAD `.to("cuda")` warning in logs**
→ Some silero-vad versions don't expose `.to()`.
  The code gracefully falls back to CPU for VAD only; Whisper still runs on GPU.
  This is harmless — VAD is fast on CPU even for large datasets.
