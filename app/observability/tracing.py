"""Per-run tracing: every pipeline step is a timed span, stored in the stage_events table.

Design decisions
────────────────
* One Trace per run attempt, held in a ContextVar. Code anywhere in the pipeline (worker, derived cache,
  faster-whisper hooks) calls `span("stage")` without being handed the trace; outside a run (tests, scripts) the
  call is a cheap no-op. ContextVars are per thread, so the in-process thread pool and Celery children each see
  only their own run.
* Spans nest. Each event records its parent, so the dashboard can draw a timeline (run > transcribe > chunk 3 >
  encoder) and compute "self time" (a span minus its children), i.e. time not explained by a finer stage.
* One clock. Durations come from perf_counter (monotonic, high resolution). The absolute start time of every span
  is the trace's wall-clock start plus its perf_counter offset, so all spans of a run share one time base and runs
  from different workers on the same machine line up on the dashboard's timeline. queue_wait is the exception: it
  starts before the worker sees the run, so it uses the database timestamps (enqueued_at -> claimed).
* Buffered, written once. Events are kept in memory and inserted in one statement when the run finishes or
  fails, right after the run's status is committed (a telemetry failure never fails a run). A span costs about a
  microsecond, against tens of milliseconds for the cheapest model call.
"""
from __future__ import annotations

import os
import socket
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Iterator

_current: ContextVar["Trace | None"] = ContextVar("asr_trace", default=None)


class Trace:
    def __init__(self, run_id: str, attempt: int = 1):
        self.run_id, self.attempt = run_id, attempt
        self.t0_wall = datetime.now(timezone.utc)
        self.t0 = time.perf_counter()
        self.worker = f"{socket.gethostname()}:{os.getpid()}"  # re-read: Celery forks after import
        self.events: list[dict] = []
        self._stack: list[int] = []

    @contextmanager
    def span(self, stage: str, **attrs) -> Iterator[dict]:
        """Time a block. Yields the span's attrs dict, so the block can add facts it learns (tokens, hit/miss)."""
        seq = len(self.events)
        ev = {"seq": seq, "parent_seq": self._stack[-1] if self._stack else None, "stage": stage,
              "attrs": attrs, "offset_s": 0.0, "duration_s": None}
        self.events.append(ev)
        self._stack.append(seq)
        t = time.perf_counter()
        ev["offset_s"] = t - self.t0
        try:
            yield attrs
        except BaseException as e:
            attrs["error"] = type(e).__name__
            raise
        finally:
            ev["duration_s"] = time.perf_counter() - t
            self._stack.pop()

    def add(self, stage: str, started_at: datetime, duration_s: float, **attrs) -> None:
        """Record a span measured elsewhere (queue wait, from database timestamps)."""
        if started_at.tzinfo is None:  # SQLite returns naive UTC
            started_at = started_at.replace(tzinfo=timezone.utc)
        self.events.append({"seq": len(self.events), "parent_seq": None, "stage": stage, "attrs": attrs,
                            "offset_s": (started_at - self.t0_wall).total_seconds(),
                            "duration_s": max(duration_s, 0.0)})

    def rows(self) -> list[dict]:
        """Rows for the stage_events table (unfinished spans, e.g. after a crash, get their time so far)."""
        now = time.perf_counter() - self.t0
        out = []
        for ev in self.events:
            dur = ev["duration_s"] if ev["duration_s"] is not None else max(now - ev["offset_s"], 0.0)
            out.append({"run_id": self.run_id, "attempt": self.attempt, "seq": ev["seq"],
                        "parent_seq": ev["parent_seq"], "stage": ev["stage"],
                        "started_at": self.t0_wall + timedelta(seconds=ev["offset_s"]),
                        "duration_ms": round(dur * 1000.0, 3), "worker": self.worker,
                        "attrs": _jsonable(ev["attrs"]) or None})
        return out

    def summary(self) -> dict[str, float]:
        """Self time in seconds per stage category (see self_times), stored in the run profile so that exports
        carry the breakdown too."""
        return {k: round(v / 1000.0, 6) for k, v in self_times(self.rows()).items()}


def stage_category(stage: str, parent_stage: str | None) -> str:
    """The encoder call made by language detection is reported apart from the transcription encoder calls."""
    if stage == "model.encode" and parent_stage == "model.language_detect":
        return "model.encode.lang"
    return stage


def self_times(rows: list[dict]) -> dict[str, float]:
    """Milliseconds per stage category, counting each span minus its children ("self time"), so the values add
    up to the run's wall time (plus queue wait) with nothing counted twice. A parent's self time is work no finer
    span explains: Python glue, generator overhead, I/O between steps."""
    by_seq = {r["seq"]: r for r in rows}
    child_ms: dict[int, float] = {}
    for r in rows:
        if r["parent_seq"] is not None:
            child_ms[r["parent_seq"]] = child_ms.get(r["parent_seq"], 0.0) + r["duration_ms"]
    out: dict[str, float] = {}
    for r in rows:
        parent = by_seq.get(r["parent_seq"]) if r["parent_seq"] is not None else None
        cat = stage_category(r["stage"], parent["stage"] if parent else None)
        out[cat] = out.get(cat, 0.0) + max(r["duration_ms"] - child_ms.get(r["seq"], 0.0), 0.0)
    return out


def _jsonable(attrs: dict) -> dict:
    out = {}
    for k, v in attrs.items():
        if isinstance(v, float):
            v = round(v, 6)
        elif not isinstance(v, (int, str, bool, type(None), list, dict)):
            v = str(v)
        out[k] = v
    return out


@contextmanager
def activate(trace: Trace) -> Iterator[Trace]:
    token = _current.set(trace)
    try:
        yield trace
    finally:
        _current.reset(token)


def current() -> Trace | None:
    return _current.get()


@contextmanager
def span(stage: str, **attrs) -> Iterator[dict]:
    """Time a block in the current run's trace; a no-op (still yields a dict) when no trace is active."""
    trace = _current.get()
    if trace is None:
        yield attrs
        return
    with trace.span(stage, **attrs) as a:
        yield a


def annotate(**attrs) -> None:
    """Add facts to the innermost open span of the current trace (no-op outside a run)."""
    trace = _current.get()
    if trace is not None and trace._stack:
        trace.events[trace._stack[-1]]["attrs"].update(attrs)
