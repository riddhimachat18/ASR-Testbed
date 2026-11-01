from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration comes from environment variables (or .env)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_root: str = "./data/blobs"
    database_url: str = "sqlite:///./data/app.db"
    storage_backend: str = "local"  # "local" now, "s3" later
    inference_sample_rate: int = 16000  # model-side constant, NOT baked into storage
    max_upload_bytes: int = 500 * 1024 * 1024
    worker_threads: int = 4
    stub_sleep_seconds: float = 0.5
    db_pool_size: int = 20
    db_max_overflow: int = 20
    db_pool_timeout: int = 30

    # --- S3 / MinIO (used when STORAGE_BACKEND=s3) ---
    s3_bucket: str = "asr-audio"
    s3_endpoint_url: str | None = None  # e.g. http://localhost:9000 for MinIO; unset for AWS
    s3_region: str = "us-east-1"
    s3_access_key_id: str | None = None  # unset = AWS default credential chain (env vars, instance role, ...)
    s3_secret_access_key: str | None = None
    s3_prefix: str = ""  # lets several environments share one bucket

    # --- queue (Part E) ---
    queue_backend: str = "inprocess"  # "inprocess" | "celery"
    redis_url: str = "redis://localhost:6379/0"
    worker_db_pool_size: int = 2  # each Celery child process runs one task at a time
    worker_db_max_overflow: int = 2
    task_soft_time_limit_s: int = 600  # task gets SoftTimeLimitExceeded -> run is marked failed cleanly
    task_hard_margin_s: int = 60  # hard kill = soft + margin
    visibility_timeout_s: int = 3600  # Redis redelivers unacked messages after this

    # --- observability ---
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"
    metrics_token: str | None = None  # if set, /metrics requires "Authorization: Bearer <token>"

    # --- recovery sweep ---
    recovery_interval_s: int = 120  # 0 disables the background sweep
    stuck_after_s: int = 900  # 'processing' for longer than this => worker presumed dead
    queued_after_s: int = 900  # 'queued' and not (re)enqueued for this long => presumed orphaned
    max_attempts: int = 3  # a run that kills its worker this many times is failed, not retried forever
    recovery_batch: int = 1000

    @property
    def task_hard_time_limit_s(self) -> int:
        return self.task_soft_time_limit_s + self.task_hard_margin_s

    @model_validator(mode="after")
    def _check_timing_invariants(self):
        hard = self.task_hard_time_limit_s
        if self.stuck_after_s <= hard:
            raise ValueError(f"STUCK_AFTER_S ({self.stuck_after_s}) must exceed the hard task limit ({hard}), "
                             "otherwise healthy long-running tasks would be reset")
        if self.visibility_timeout_s <= hard:
            raise ValueError(f"VISIBILITY_TIMEOUT_S ({self.visibility_timeout_s}) must exceed the hard task limit ({hard})")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
