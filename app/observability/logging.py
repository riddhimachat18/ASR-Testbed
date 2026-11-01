"""Structured logging: one JSON object per line, with run_id / request_id attached automatically.

    with log_context(run_id=run_id):        # every log line emitted inside (any module, any library) carries run_id
        log_event(log, "run.done", latency_s=1.2, wer=0.05)

Context lives in contextvars, so it follows a request into FastAPI's worker threads and stays isolated per thread.
It is attached by a LogRecord factory, which means it works with every handler (including pytest's caplog).
Do not pass `extra={"run_id": ...}` yourself: use log_context. Event fields go through log_event(**fields).
"""
import json
import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone

_ctx: ContextVar[dict] = ContextVar("asr_log_ctx", default={})
_RESERVED = set(vars(logging.makeLogRecord({}))) | {"message", "asctime", "fields"}
_factory_installed = False


@contextmanager
def log_context(**fields):
    token = _ctx.set({**_ctx.get(), **{k: v for k, v in fields.items() if v is not None}})
    try:
        yield
    finally:
        _ctx.reset(token)


def current_context() -> dict:
    return dict(_ctx.get())


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, exc_info=None, **fields) -> None:
    """Emit a named event with structured fields (the event name is the log message)."""
    logger.log(level, event, exc_info=exc_info, extra={"fields": fields})


def _install_record_factory() -> None:
    global _factory_installed
    if _factory_installed:
        return
    previous = logging.getLogRecordFactory()

    def factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        for key, value in _ctx.get().items():
            setattr(record, key, value)
        return record

    logging.setLogRecordFactory(factory)
    _factory_installed = True


def _extras(record: logging.LogRecord) -> dict:
    out = {k: v for k, v in record.__dict__.items() if k not in _RESERVED and not k.startswith("_")}
    fields = getattr(record, "fields", None)
    if isinstance(fields, dict):
        out.update(fields)
    return out


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "pid": record.process,
        }
        for key, value in _extras(record).items():
            out.setdefault(key, value)
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    def __init__(self):
        super().__init__("%(asctime)s %(levelname)-7s %(name)s %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = " ".join(f"{k}={v}" for k, v in _extras(record).items())
        return f"{base} {extras}".rstrip()


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Idempotent. Only manages its own handler on the root logger; foreign handlers are left alone."""
    _install_record_factory()
    formatter = JsonFormatter() if fmt == "json" else TextFormatter()
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_asr_handler", False):
            root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler._asr_handler = True
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level.upper())
    # uvicorn configures its own handlers before our app is imported: give them the same formatter.
    # Its access log is replaced by our request middleware (which adds route, duration, request_id).
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        for h in logging.getLogger(name).handlers:
            h.setFormatter(formatter)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
