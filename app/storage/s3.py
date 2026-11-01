"""S3-compatible storage (AWS S3, MinIO, ...) behind the same StorageBackend interface as local disk.

Contract parity with LocalDiskStorage:
  * immutable blobs: writes use a conditional PUT (If-None-Match: *), so two writers can never both win and nothing
    is ever overwritten, even under a race. (Needs a server that supports conditional writes: AWS S3 since 2024,
    MinIO, Ceph RGW.)
  * streaming in, streaming out, constant memory: uploads are multipart, one part in memory at a time; the SHA-256
    is computed on the fly. An interrupted upload is aborted, so no partial blob and no orphaned parts remain.
  * get_audio returns a SEEKABLE stream (soundfile, WAV headers etc. need seek). It is backed by ranged GETs with a
    read-ahead buffer, so reading a header or a slice does not download the whole object.
  * the same key rules (validate_key) and the same exceptions.
A key prefix lets several environments share one bucket; it is invisible to callers.
"""
import hashlib
import io
import errno
import logging
from typing import BinaryIO, Iterator

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .base import BlobExistsError, BlobInfo, BlobNotFoundError, StorageBackend, validate_key

log = logging.getLogger("storage.s3")

MIN_PART = 5 * 1024 * 1024        # S3's minimum size for every part except the last
PARTS_PER_STEP = 1000             # part size grows every 1000 parts so 10,000 parts reach ~275 GB at the minimum


def _http_status(e: ClientError) -> int | None:
    return e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")


def _code(e: ClientError) -> str:
    return e.response.get("Error", {}).get("Code", "")


def _is_missing(e: ClientError) -> bool:
    return _http_status(e) == 404 and _code(e) not in ("NoSuchBucket",) or _code(e) in ("404", "NoSuchKey", "NotFound")


def _is_conflict(e: ClientError) -> bool:
    return _http_status(e) in (409, 412) or _code(e) in ("PreconditionFailed", "ConditionalRequestConflict")


def _read_full(stream, n: int) -> bytes:
    """Read up to n bytes, looping over short reads; returns fewer than n only at end of stream."""
    data = stream.read(n)
    if len(data) == n or not data:
        return data
    parts, got = [data], len(data)
    while got < n:
        more = stream.read(n - got)
        if not more:
            break
        parts.append(more)
        got += len(more)
    return b"".join(parts)


class _S3RangeReader(io.RawIOBase):
    """Seekable raw reader over one object, one ranged GET per read call (wrapped in a BufferedReader)."""

    def __init__(self, client, bucket: str, key: str, size: int):
        super().__init__()
        self._client, self._bucket, self._key, self._size, self._pos = client, bucket, key, size, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}.get(whence)
        if base is None:
            raise ValueError(f"invalid whence: {whence}")
        if base + offset < 0:
            raise OSError(errno.EINVAL, "negative seek position")
        self._pos = base + offset
        return self._pos

    def readinto(self, b) -> int:
        if self._pos >= self._size or len(b) == 0:
            return 0
        want = min(len(b), self._size - self._pos)
        resp = self._client.get_object(Bucket=self._bucket, Key=self._key,
                                       Range=f"bytes={self._pos}-{self._pos + want - 1}")
        try:
            data = resp["Body"].read()
        finally:
            resp["Body"].close()
        b[:len(data)] = data
        self._pos += len(data)
        return len(data)


class S3Storage(StorageBackend):
    def __init__(self, bucket: str, *, prefix: str = "", endpoint_url: str | None = None,
                 region: str = "us-east-1", access_key_id: str | None = None,
                 secret_access_key: str | None = None, part_size: int = MIN_PART,
                 read_buffer: int = 4 * 1024 * 1024, client=None):
        if part_size < MIN_PART:
            raise ValueError(f"part_size must be at least {MIN_PART} bytes (S3 minimum)")
        self.bucket, self.part_size, self.read_buffer, self.region = bucket, part_size, read_buffer, region
        self.prefix = prefix if not prefix or prefix.endswith("/") else prefix + "/"
        self._client = client or boto3.client(
            "s3", endpoint_url=endpoint_url, region_name=region,
            aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key,   # None = default credential chain
            config=Config(
                retries={"max_attempts": 5, "mode": "standard"}, max_pool_connections=50,
                connect_timeout=5, read_timeout=60,
                s3={"addressing_style": "path" if endpoint_url else "auto"},            # MinIO needs path style
                # boto3 >= 1.36 adds CRC32 trailers by default, which some S3-compatible servers reject.
                request_checksum_calculation="when_required", response_checksum_validation="when_required",
            ),
        )

    # ---------------------------------------------------------------- helpers
    def _full(self, key: str) -> str:
        return self.prefix + validate_key(key)

    def _head(self, full: str):
        try:
            return self._client.head_object(Bucket=self.bucket, Key=full)
        except ClientError as e:
            if _is_missing(e):
                return None
            raise

    def _part_size(self, part_number: int) -> int:
        return self.part_size * (1 + (part_number - 1) // PARTS_PER_STEP)

    def ensure_bucket(self) -> None:
        """Create the bucket if it does not exist (development and tests; production buckets are provisioned)."""
        try:
            self._client.head_bucket(Bucket=self.bucket)
        except ClientError as e:
            if _http_status(e) not in (404, 301) and _code(e) not in ("404", "NoSuchBucket"):
                raise
            kwargs = {} if self.region == "us-east-1" else {"CreateBucketConfiguration": {"LocationConstraint": self.region}}
            self._client.create_bucket(Bucket=self.bucket, **kwargs)

    # ---------------------------------------------------------------- StorageBackend
    def save_audio(self, key: str, stream: BinaryIO) -> BlobInfo:
        full = self._full(key)
        if self._head(full) is not None:           # cheap early exit; the conditional write below is the real guard
            raise BlobExistsError(key)
        sha = hashlib.sha256()
        first = _read_full(stream, self._part_size(1))
        sha.update(first)
        try:
            if len(first) < self._part_size(1):    # whole object fits in one part: a single conditional PUT
                self._client.put_object(Bucket=self.bucket, Key=full, Body=first, IfNoneMatch="*")
                size = len(first)
            else:
                size = self._multipart(full, first, stream, sha)
        except ClientError as e:
            if _is_conflict(e):
                raise BlobExistsError(key) from None
            raise
        return BlobInfo(key=key, size_bytes=size, sha256=sha.hexdigest())

    def _multipart(self, full: str, first: bytes, stream, sha) -> int:
        upload_id = self._client.create_multipart_upload(Bucket=self.bucket, Key=full)["UploadId"]
        parts, size, number, chunk = [], 0, 1, first
        try:
            while chunk:
                body, chunk = chunk, b""           # drop our reference so only ~one part is alive at a time
                resp = self._client.upload_part(Bucket=self.bucket, Key=full, UploadId=upload_id,
                                                PartNumber=number, Body=body)
                parts.append({"PartNumber": number, "ETag": resp["ETag"]})
                size += len(body)
                del body
                number += 1
                chunk = _read_full(stream, self._part_size(number))
                sha.update(chunk)
            self._client.complete_multipart_upload(Bucket=self.bucket, Key=full, UploadId=upload_id,
                                                   MultipartUpload={"Parts": parts}, IfNoneMatch="*")
        except BaseException:
            try:
                self._client.abort_multipart_upload(Bucket=self.bucket, Key=full, UploadId=upload_id)
            except Exception:  # noqa: BLE001
                log.exception("could not abort multipart upload %s for %s", upload_id, full)
            raise
        return size

    def get_audio(self, key: str) -> BinaryIO:
        full = self._full(key)
        head = self._head(full)
        if head is None:
            raise BlobNotFoundError(key)
        raw = _S3RangeReader(self._client, self.bucket, full, head["ContentLength"])
        return io.BufferedReader(raw, buffer_size=self.read_buffer)

    def exists(self, key: str) -> bool:
        return self._head(self._full(key)) is not None

    def list_keys(self, prefix: str = "") -> Iterator[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix + prefix):
            for obj in page.get("Contents", []):
                yield obj["Key"][len(self.prefix):]

    def delete(self, key: str) -> bool:
        full = self._full(key)
        if self._head(full) is None:
            return False
        self._client.delete_object(Bucket=self.bucket, Key=full)
        return True
