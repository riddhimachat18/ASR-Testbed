from typing import Iterator

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session

from app.db.models import User
from app.db.repositories import UserRepo
from app.services.audio_service import AudioService


def get_db(request: Request) -> Iterator[Session]:
    with request.app.state.session_factory() as s:
        yield s


def get_current_user(x_api_key: str | None = Header(default=None), db: Session = Depends(get_db)) -> User:
    user = UserRepo(db).get_by_api_key(x_api_key) if x_api_key else None
    if user is None:
        raise HTTPException(401, "Missing or invalid API key")
    db.commit()
    return user


def get_service(request: Request, db: Session = Depends(get_db)) -> AudioService:
    st = request.app.state
    return AudioService(db, st.storage, st.registry, st.settings)
