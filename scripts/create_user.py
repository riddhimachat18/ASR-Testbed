"""Usage: python -m scripts.create_user <username>   (run `alembic upgrade head` first)"""
import sys

from app.config import get_settings
from app.db.repositories import UserRepo
from app.db.session import make_engine, make_session_factory

if __name__ == "__main__":
    engine = make_engine(get_settings().database_url)
    with make_session_factory(engine)() as s:
        user, key = UserRepo(s).create(sys.argv[1])
        s.commit()
    print(f"user={user.username} id={user.id}\nAPI key (shown once): {key}")
