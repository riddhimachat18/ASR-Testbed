import json, sys
from app.config import get_settings
from app.db.repositories import UserRepo
from app.db.session import make_engine, make_session_factory

n = int(sys.argv[1])
engine = make_engine(get_settings().database_url)
keys = []
with make_session_factory(engine)() as s:
    for i in range(n):
        _, key = UserRepo(s).create(f"loaduser{i}")
        keys.append(key)
    s.commit()
json.dump(keys, open("keys.json", "w"))