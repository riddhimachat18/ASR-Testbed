# GPU Baseline Results

Same format as `README.md` — the GPU results are stored alongside the CPU ones
so they can be compared side-by-side.

## Naming convention

```
baseline_gpu_<model-version>_<YYYYMMDDTHHMMSS>.json
```

The `gpu_` prefix distinguishes GPU runs from CPU runs so they sort clearly in
the `results/` directory.

## Expected ranges (NVIDIA GPU, e.g. RTX 3080 / A100)

| Model              | WER (test-clean) | RTF (GPU, float16) |
|--------------------|------------------|--------------------|
| faster-whisper-tiny  | ~5–8%          | ~0.03–0.08x        |
| faster-whisper-base  | ~4–6%          | ~0.05–0.12x        |
| faster-whisper-small | ~3–4%          | ~0.08–0.20x        |

RTF < 1.0 = faster than real-time.
GPU RTF is typically 10–50× lower than CPU RTF.

## CPU baseline for comparison

| Model              | WER   | RTF (CPU, MacBook) |
|--------------------|-------|--------------------|
| faster-whisper-tiny | 9.33% | 1.583x             |

The CPU baseline JSON is in `results/CPU/baseline_faster-whisper-tiny_20261101T105820.json`.

## How to generate a GPU baseline report

```bash
# Run the GPU server
uvicorn gpu.main_gpu:factory --factory --port 8001

# Submit and score the dataset (same scripts as CPU, just point at port 8001)
python scripts/submit_dataset.py --url http://localhost:8001 ...

# Generate the baseline JSON — save directly into results/GPU/
python scripts/report_baseline.py --output results/GPU/baseline_faster-whisper-tiny_$(date +%Y%m%dT%H%M%S).json
```

## Comparing CPU vs GPU results

```bash
# CPU baseline lives in results/CPU/
# GPU baseline lives in results/GPU/

python scripts/report_baseline.py \
    results/CPU/baseline_faster-whisper-tiny_*.json \
    results/GPU/baseline_faster-whisper-tiny_*.json
```

Key metrics: **WER** (should be equal or better on GPU with float16) and **RTF** (should be 10–50× lower on GPU).
