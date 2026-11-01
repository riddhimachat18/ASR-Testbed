from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import BinaryIO, Iterator


class StorageError(Exception): ...
class BlobExistsError(StorageError): ...
class BlobNotFoundError(StorageError): ...
class InvalidKeyError(StorageError): ...


@dataclass(frozen=True)
class BlobInfo:
    key: str
    size_bytes: int
    sha256: str


class StorageBackend(ABC):
    """The only thing the API/pipeline ever see. Blobs are immutable."""

    @abstractmethod
    def save_audio(self, key: str, stream: BinaryIO) -> BlobInfo:
        """Stream `stream` into `key`. Raises BlobExistsError if key exists."""

    @abstractmethod
    def get_audio(self, key: str) -> BinaryIO:
        """Open a seekable binary stream. Raises BlobNotFoundError."""

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def list_keys(self, prefix: str = "") -> Iterator[str]: ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Returns True if something was deleted."""


def validate_key(key: str) -> str:
    """One key contract for every backend, so a key valid on disk is valid in S3 and vice versa.
    Keys are relative, '/'-separated, with no empty, '.' or '..' segments, no backslashes or control characters."""
    if (not key or len(key.encode("utf-8", "replace")) > 1024 or key.startswith("/") or "\\" in key
            or any(ord(c) < 32 for c in key)):
        raise InvalidKeyError(key)
    if any(part in ("", ".", "..") for part in key.split("/")):
        raise InvalidKeyError(key)
    return key
