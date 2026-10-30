from functools import lru_cache

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
