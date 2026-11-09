"""Register existing keys.json API keys into the configured database (Postgres).

Run with:
    python scripts/register_keys_pg.py
"""
import hashlib, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select
from app.db.models import User
from app.db.session import make_engine, make_session_factory

# GPUSettings reads .env.gpu so DATABASE_URL points at Postgres
from gpu.config_gpu import get_gpu_settings

settings = get_gpu_settings()
engine   = make_engine(settings.database_url)
sf       = make_session_factory(engine)
keys     = json.loads(Path("keys.json").read_text())

with sf() as s:
    existing_hashes = {u.hashed_api_key for u in s.scalars(select(User)).all()}
    added = 0
    for i, key in enumerate(keys):
        h = hashlib.sha256(key.encode()).hexdigest()
        if h not in existing_hashes:
            s.add(User(username=f"loadtest_user_{i}", hashed_api_key=h))
            added += 1
    s.commit()
    total = s.execute(select(User)).fetchall()
    print(f"Added {added} users  |  Total in DB: {len(total)}")
