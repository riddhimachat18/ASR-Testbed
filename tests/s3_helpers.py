"""Shared helpers for S3/MinIO tests.

Configuration is read from environment variables (defaulting to the values
defined in docker-compose.yml):

    S3_ENDPOINT_URL      http://localhost:9000
    S3_ACCESS_KEY_ID     minioadmin
    S3_SECRET_ACCESS_KEY minioadmin
    S3_BUCKET            asr-test-<uuid>   (created fresh per test session)

Any test that calls new_s3_storage() or require_s3() is automatically skipped
when MinIO is not reachable, so the suite degrades gracefully without Docker.
"""

import os
import uuid

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError, EndpointResolutionError

from app.storage.s3 import S3Storage

# ------------------------------------------------------------------ config
_ENDPOINT = os.environ.get("S3_ENDPOINT_URL", "http://localhost:9000")
_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY_ID", "minioadmin")
_SECRET_KEY = os.environ.get("S3_SECRET_ACCESS_KEY", "minioadmin")
_BUCKET = os.environ.get("S3_TEST_BUCKET", f"asr-test-{uuid.uuid4().hex[:8]}")


def _s3_config() -> dict:
    """Return kwargs suitable for S3Storage(...) pointing at the test MinIO."""
    return dict(
        bucket=_BUCKET,
        endpoint_url=_ENDPOINT,
        access_key_id=_ACCESS_KEY,
        secret_access_key=_SECRET_KEY,
    )


def _check_reachable() -> str | None:
    """Return None if MinIO is reachable, or a skip reason string if not."""
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        endpoint_url=_ENDPOINT,
        aws_access_key_id=_ACCESS_KEY,
        aws_secret_access_key=_SECRET_KEY,
        region_name="us-east-1",
        config=Config(
            connect_timeout=2,
            read_timeout=2,
            retries={"max_attempts": 1},
            s3={"addressing_style": "path"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )
    try:
        client.list_buckets()
        return None
    except (ClientError, EndpointConnectionError, EndpointResolutionError, OSError):
        return f"MinIO not reachable at {_ENDPOINT} (run: docker compose up -d minio)"


# Cached at import time so every call in one process pays the cost once.
_skip_reason: str | None | bool = False  # False = not yet checked


def _get_skip_reason() -> str | None:
    global _skip_reason
    if _skip_reason is False:
        _skip_reason = _check_reachable()
    return _skip_reason  # type: ignore[return-value]


# ------------------------------------------------------------------ public API

def require_s3() -> dict:
    """Return S3 connection kwargs, or pytest.skip() if MinIO is not available.

    Use this inside test functions or fixtures when you need the raw kwargs
    (e.g. to construct a second S3Storage with a different prefix).
    """
    reason = _get_skip_reason()
    if reason:
        pytest.skip(reason)
    return _s3_config()


def new_s3_storage() -> S3Storage:
    """Return a ready-to-use S3Storage pointed at MinIO, or pytest.skip().

    The bucket is created if it does not exist.  Call wipe(st) in your fixture
    teardown to clean up all objects written during the test.
    """
    cfg = require_s3()  # skips if unreachable
    st = S3Storage(**cfg)
    st.ensure_bucket()
    return st


def wipe(storage: S3Storage) -> None:
    """Delete every object in storage (used in fixture teardown)."""
    for key in list(storage.list_keys()):
        storage.delete(key)
