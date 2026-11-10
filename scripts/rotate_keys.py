"""Replace every user's API key, in place (users own audio rows, so they are never deleted).

Why: keys.json and an API key in .env were pushed to GitHub before they were ignored. The history has been
rewritten, but GitHub still serves the old commits through PR #1, so the only real fix is to make the leaked
keys useless. Run this once on every database (the CPU machine's and the GPU machine's).

    python -m scripts.rotate_keys            # asks for confirmation
    python -m scripts.rotate_keys --yes

Writes the new load-test keys (users whose name starts with one of --prefix) to keys.json, which is git-ignored,
and prints the new key of every other user once. Put the one you use by hand into .env as KEY=...
"""
import argparse
import json
import secrets
from pathlib import Path

from sqlalchemy import select

from app.config import get_settings
from app.db.models import User
from app.db.repositories import hash_api_key
from app.db.session import make_engine, make_session_factory

LOAD_TEST_PREFIXES = ("loaduser", "loadtest_user_")  # create_users.py and the old register_keys_pg.py


def rotate(session_factory, prefixes: tuple[str, ...]) -> tuple[list[str], dict[str, str]]:
    """Give every user a new key. Returns (load-test keys, {username: key} for everyone else)."""
    load_keys: list[str] = []
    others: dict[str, str] = {}
    with session_factory() as s:
        for user in s.scalars(select(User).order_by(User.created_at, User.username)):
            key = secrets.token_urlsafe(32)  # same generator as UserRepo.create
            user.hashed_api_key = hash_api_key(key)
            if user.username.startswith(prefixes):
                load_keys.append(key)
            else:
                others[user.username] = key
        s.commit()
    return load_keys, others


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="keys.json", help="where to write the load-test keys (git-ignored)")
    ap.add_argument("--prefix", action="append", help="username prefix of load-test users (repeatable)")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    a = ap.parse_args()

    st = get_settings()
    sf = make_session_factory(make_engine(st.database_url))
    with sf() as s:
        n = len(s.scalars(select(User.id)).all())
    if not a.yes and input(f"Replace the API keys of all {n} users in {st.database_url.split('@')[-1]}? [y/N] ").lower() != "y":
        print("nothing changed")
        return
    load_keys, others = rotate(sf, tuple(a.prefix or LOAD_TEST_PREFIXES))
    Path(a.out).write_text(json.dumps(load_keys))
    print(f"rotated {len(load_keys) + len(others)} keys; {len(load_keys)} load-test keys written to {a.out}")
    for username, key in others.items():
        print(f"  {username}: {key}")
    if others:
        print("These are shown once. Put the one you use into .env as KEY=...")


if __name__ == "__main__":
    main()
