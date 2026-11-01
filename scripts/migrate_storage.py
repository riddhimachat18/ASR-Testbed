"""Copy blobs from one storage backend to another, e.g. when moving from local disk to MinIO/S3.

    python -m scripts.migrate_storage --from local --to s3 --dry-run
    python -m scripts.migrate_storage --from local --to s3 [--prefix audio/] [--no-verify] [--create-bucket]

Safe to repeat (existing keys are skipped and, unless --no-verify, checked by SHA-256). Nothing is deleted from the
source. Afterwards set STORAGE_BACKEND=s3 and restart the API and workers. Derived artifacts (derived/) are rebuilt on
demand, so `--prefix audio/` is enough if you do not want to copy them.
"""
import argparse
import sys

from app.config import get_settings
from app.storage.factory import build_storage
from app.storage.transfer import copy_blobs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True, choices=["local", "s3"])
    ap.add_argument("--to", dest="dst", required=True, choices=["local", "s3"])
    ap.add_argument("--prefix", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--create-bucket", action="store_true")
    a = ap.parse_args()
    if a.src == a.dst:
        sys.exit("--from and --to must differ")
    st = get_settings()
    src, dst = build_storage(a.src, st), build_storage(a.dst, st)
    if a.create_bucket and a.dst == "s3":
        dst.ensure_bucket()
    r = copy_blobs(src, dst, prefix=a.prefix, dry_run=a.dry_run, verify=not a.no_verify)
    if a.dry_run:
        print(f"would copy {len(r.planned)} blobs; {r.skipped_existing} already at the destination")
    else:
        print(f"copied {r.copied} blobs ({r.bytes_copied / 1e6:.1f} MB); {r.skipped_existing} already present")
    if r.mismatched:
        print(f"CONTENT MISMATCH for {len(r.mismatched)} keys (destination differs from source):")
        for k in r.mismatched:
            print("  ", k)
        sys.exit(1)
