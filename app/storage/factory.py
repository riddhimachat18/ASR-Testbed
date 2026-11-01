from app.config import Settings

from .base import StorageBackend
from .local import LocalDiskStorage


def build_storage(backend: str, settings: Settings) -> StorageBackend:
    if backend == "local":
        return LocalDiskStorage(settings.data_root)
    if backend == "s3":
        from .s3 import S3Storage  # imported lazily: boto3 is only needed when S3 is in use
        if not settings.s3_bucket:
            raise ValueError("STORAGE_BACKEND=s3 requires S3_BUCKET")
        return S3Storage(settings.s3_bucket, prefix=settings.s3_prefix, endpoint_url=settings.s3_endpoint_url,
                         region=settings.s3_region, access_key_id=settings.s3_access_key_id,
                         secret_access_key=settings.s3_secret_access_key)
    raise ValueError(f"Unknown storage backend: {backend!r} (expected 'local' or 's3')")


def get_storage(settings: Settings) -> StorageBackend:
    return build_storage(settings.storage_backend, settings)
