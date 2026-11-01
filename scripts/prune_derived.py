"""Delete cached derived artifacts whose original audio no longer exists.

    python -m scripts.prune_derived --dry-run     # list what would be removed
    python -m scripts.prune_derived               # remove it
"""
import argparse

from app.config import get_settings
from app.db.session import make_engine, make_session_factory
from app.pipeline.derived import prune_orphans
from app.storage.factory import get_storage

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    st = get_settings()
    sf = make_session_factory(make_engine(st.database_url, st.db_pool_size, st.db_max_overflow, st.db_pool_timeout))
    removed = prune_orphans(get_storage(st), sf, dry_run=a.dry_run)
    print(f"{'would remove' if a.dry_run else 'removed'} {len(removed)} orphaned derived artifacts")
    for k in removed:
        print("  ", k)
