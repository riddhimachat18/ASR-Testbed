"""Copy blobs between backends (e.g. local disk -> S3). Possible because the database stores only relative keys:
after the copy, switching STORAGE_BACKEND is the whole migration."""
import hashlib
from dataclasses import dataclass, field

from .base import StorageBackend


@dataclass
class TransferReport:
    copied: int = 0
    skipped_existing: int = 0
    bytes_copied: int = 0
    planned: list[str] = field(default_factory=list)      # dry run only
    mismatched: list[str] = field(default_factory=list)   # key exists at the destination with different content


def sha256_of(storage: StorageBackend, key: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with storage.get_audio(key) as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def copy_blobs(src: StorageBackend, dst: StorageBackend, prefix: str = "", dry_run: bool = False,
               verify: bool = True) -> TransferReport:
    """Idempotent: keys already at the destination are skipped (and, with verify, compared by SHA-256)."""
    report = TransferReport()
    for key in src.list_keys(prefix):
        if dst.exists(key):
            if verify and sha256_of(src, key) != sha256_of(dst, key):
                report.mismatched.append(key)
            else:
                report.skipped_existing += 1
            continue
        if dry_run:
            report.planned.append(key)
            continue
        with src.get_audio(key) as f:
            info = dst.save_audio(key, f)
        if verify and sha256_of(dst, key) != info.sha256:   # read back what actually landed
            report.mismatched.append(key)
            continue
        report.copied += 1
        report.bytes_copied += info.size_bytes
    return report
