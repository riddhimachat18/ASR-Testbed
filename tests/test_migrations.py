from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext

from app.config import get_settings
from app.db.models import Base
from app.db.session import make_engine

ROOT = Path(__file__).resolve().parent.parent


def _cfg():
    return Config(str(ROOT / "alembic.ini"))


def test_migrations_apply_roll_back_and_match_the_models(tmp_path, monkeypatch):
    """Fails if someone changes a model but forgets the migration (or the migration is broken)."""
    url = f"sqlite:///{tmp_path / 'm.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    get_settings.cache_clear()
    try:
        command.upgrade(_cfg(), "head")
        engine = make_engine(url)
        with engine.connect() as conn:
            assert compare_metadata(MigrationContext.configure(conn), Base.metadata) == []
        engine.dispose()
        command.downgrade(_cfg(), "base")
        command.upgrade(_cfg(), "head")
    finally:
        get_settings.cache_clear()
