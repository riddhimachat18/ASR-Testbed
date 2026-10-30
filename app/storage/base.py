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
