import uuid
from datetime import datetime, timezone


def new_audio_key(ext: str, now: datetime | None = None) -> str:
    """Relative, date-sharded, UUID-named: audio/2026-10-23/<uuid>.wav"""
    now = now or datetime.now(timezone.utc)
    return f"audio/{now:%Y-%m-%d}/{uuid.uuid4().hex}{ext.lower()}"
