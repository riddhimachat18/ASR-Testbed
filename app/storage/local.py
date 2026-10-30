import hashlib
import os
import uuid
from pathlib import Path
from typing import BinaryIO, Iterator

from .base import (BlobExistsError, BlobInfo, BlobNotFoundError,
                   InvalidKeyError, StorageBackend)

TMP_SUFFIX = ".tmp"


class LocalDiskStorage(StorageBackend):
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if not key or key.startswith("/") or "\\" in key:
            raise InvalidKeyError(key)
        p = (self.root / key).resolve()
        if self.root not in p.parents:  # blocks ../ traversal
            raise InvalidKeyError(key)
        return p

    def save_audio(self, key: str, stream: BinaryIO, chunk_size: int = 1 << 20) -> BlobInfo:
        dest = self._path(key)
        if dest.exists():
            raise BlobExistsError(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f"{dest.name}.{uuid.uuid4().hex}{TMP_SUFFIX}")
        sha, size = hashlib.sha256(), 0
        try:
            with open(tmp, "wb") as f:
                while chunk := stream.read(chunk_size):
                    sha.update(chunk)
                    f.write(chunk)
                    size += len(chunk)
            try:
                os.link(tmp, dest)  # atomic, refuses to overwrite
            except FileExistsError:
                raise BlobExistsError(key) from None
        finally:
            tmp.unlink(missing_ok=True)
        return BlobInfo(key=key, size_bytes=size, sha256=sha.hexdigest())

    def get_audio(self, key: str) -> BinaryIO:
        p = self._path(key)
        if not p.is_file():
            raise BlobNotFoundError(key)
        return open(p, "rb")

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list_keys(self, prefix: str = "") -> Iterator[str]:
        for p in sorted(self.root.rglob("*")):
            if p.is_file() and not p.name.endswith(TMP_SUFFIX):
                key = p.relative_to(self.root).as_posix()
                if key.startswith(prefix):
                    yield key

    def delete(self, key: str) -> bool:
        p = self._path(key)
        if p.is_file():
            p.unlink()
            return True
        return False
