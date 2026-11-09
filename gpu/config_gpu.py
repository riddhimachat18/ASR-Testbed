"""GPU-specific configuration.

Identical to app/config.py with two additions:
  - device      : "cuda" (change to "cpu" to fall back to CPU for debugging)
  - compute_type: "float16" (GPU-optimised quantization; use "int8_float16" to save VRAM)

Every other setting is unchanged so the GPU run and the CPU run are directly comparable.
Usage:
    from gpu.config_gpu import get_gpu_settings
    settings = get_gpu_settings()
"""
from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class GPUSettings(BaseSettings):
    """All configuration comes from environment variables (or .env.gpu).

    Two new fields compared with the CPU Settings:
        DEVICE       — "cuda" | "cpu" | "auto"
        COMPUTE_TYPE — "float16" | "bfloat16" | "int8_float16" | "int8"
    """

    model_config = SettingsConfigDict(env_file=".env.gpu", extra="ignore")

    data_root: str = "./data/blobs"
    database_url: str = "sqlite:///./data/app.db"
    storage_backend: str = "local"
    inference_sample_rate: int = 16000
    max_upload_bytes: int = 500 * 1024 * 1024
    worker_threads: int = 4
    stub_sleep_seconds: float = 0.5
    db_pool_size: int = 20
    db_max_overflow: int = 20
    db_pool_timeout: int = 30

    # --- S3 / MinIO (used when STORAGE_BACKEND=s3) ---
    s3_bucket: str = "asr-audio"
    s3_endpoint_url: str | None = None
    s3_region: str = "us-east-1"
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_prefix: str = ""

    # --- queue ---
    queue_backend: str = "inprocess"
    redis_url: str = "redis://localhost:6379/0"
    worker_db_pool_size: int = 2
    worker_db_max_overflow: int = 2
    task_soft_time_limit_s: int = 600
    task_hard_margin_s: int = 60
    visibility_timeout_s: int = 3600

    # --- observability ---
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"
    metrics_token: str | None = None

    # --- recovery sweep ---
    recovery_interval_s: int = 120
    stuck_after_s: int = 900
    queued_after_s: int = 900
    max_attempts: int = 3
    recovery_batch: int = 1000

    # ── GPU-specific settings (new) ────────────────────────────────────────────
    # device: "cuda" to use the GPU, "cpu" for CPU fallback, "auto" lets
    #         CTranslate2 pick (falls back to CPU if no CUDA device is found).
    device: str = "cuda"

    # compute_type:
    #   "float16"       — fastest on modern GPUs (Ampere+), recommended default
    #   "bfloat16"      — slightly slower, wider dynamic range (good for A100/H100)
    #   "int8_float16"  — INT8 weights + FP16 compute: saves VRAM, minimal quality loss
    #   "int8"          — CPU-only quantization (keep for CPU fallback / debugging)
    compute_type: str = "float16"

    @property
    def task_hard_time_limit_s(self) -> int:
        return self.task_soft_time_limit_s + self.task_hard_margin_s

    @model_validator(mode="after")
    def _check_timing_invariants(self):
        hard = self.task_hard_time_limit_s
        if self.stuck_after_s <= hard:
            raise ValueError(
                f"STUCK_AFTER_S ({self.stuck_after_s}) must exceed the hard task limit ({hard})"
            )
        if self.visibility_timeout_s <= hard:
            raise ValueError(
                f"VISIBILITY_TIMEOUT_S ({self.visibility_timeout_s}) must exceed the hard task limit ({hard})"
            )
        return self


@lru_cache
def get_gpu_settings() -> GPUSettings:
    return GPUSettings()
