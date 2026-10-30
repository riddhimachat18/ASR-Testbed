from app.config import Settings

from .base import StorageBackend
from .local import LocalDiskStorage


def get_storage(settings: Settings) -> StorageBackend:
    if settings.storage_backend == "local":
        return LocalDiskStorage(settings.data_root)
    # if settings.storage_backend == "s3": return S3Storage(...)  # MinIO step
    raise ValueError(f"Unknown STORAGE_BACKEND: {settings.storage_backend}")
