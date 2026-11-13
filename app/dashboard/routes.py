"""HTTP for the dashboard: one page (static/index.html) and read-only JSON endpoints.

  GET /dashboard                     the page
  GET /dashboard/api/groups          comparable run groups (this database + committed exports)
  GET /dashboard/api/runs?group=     one row per run: timings, encoder/decoder split, decoder confidence, features
  GET /dashboard/api/stages?group=   where the time goes: self time per stage, per-call p50/p95
  GET /dashboard/api/run/{id}        one run: its span timeline, transcript vs reference, features
  GET /dashboard/api/stream          server-sent events, one message per second: new stage events, finished runs
                                     and queue counts (the live feed). ?once=1 sends one message and closes.
If DASHBOARD_TOKEN is set, every request needs ?token=<it> (the page passes its own URL's token on).
"""
import asyncio
import json
import secrets
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from app.dashboard.data import MAX_GROUPS, DashboardData

STATIC = Path(__file__).parent / "static"


def _data(request: Request) -> DashboardData:
    st = request.app.state
    if getattr(st, "dashboard_data", None) is None:
        st.dashboard_data = DashboardData(st.session_factory, st.settings.dashboard_exports_dir)
    return st.dashboard_data


def _auth(request: Request, token: str | None = Query(default=None)) -> None:
    settings = request.app.state.settings
    if not settings.dashboard_enabled:
        raise HTTPException(404)
    expected = settings.dashboard_token
    if expected and not (token and secrets.compare_digest(token, expected)):
        raise HTTPException(401, "dashboard token required: open /dashboard?token=...")


router = APIRouter(prefix="/dashboard", dependencies=[Depends(_auth)], include_in_schema=False)


@router.get("")
def page():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@router.get("/static/{name}")
def static(name: str):
    path = STATIC / name
    if path.parent != STATIC or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, headers={"Cache-Control": "max-age=86400"})


@router.get("/api/groups")
def groups(request: Request):
    return _data(request).groups()


@router.get("/api/runs")
def runs(request: Request, group: list[str] = Query(...)):
    if len(group) > MAX_GROUPS:
        raise HTTPException(400, f"at most {MAX_GROUPS} groups at once")
    data, out = _data(request), {}
    for g in group:
        try:
            out[g] = data.rows(g)
        except (KeyError, ValueError):
            raise HTTPException(404, f"unknown group {g}")
    return JSONResponse(out)


@router.get("/api/stages")
def stages(request: Request, group: list[str] = Query(...)):
    if len(group) > MAX_GROUPS:
        raise HTTPException(400, f"at most {MAX_GROUPS} groups at once")
    data = _data(request)
    try:
        return {g: data.stages(g) for g in group}
    except (KeyError, ValueError):
        raise HTTPException(404, "unknown group")


@router.get("/api/run/{run_id}")
def run(request: Request, run_id: str):
    detail = _data(request).run_detail(run_id)
    if detail is None:
        raise HTTPException(404)
    return detail


@router.get("/api/stream")
async def stream(request: Request, since: int | None = None, once: bool = False, interval: float = 1.0):
    data = _data(request)
    interval = min(max(interval, 0.25), 10.0)

    async def gen():
        last = since
        while True:
            payload = await run_in_threadpool(data.live, last)
            last = payload["last_id"]
            yield f"data: {json.dumps(payload, default=str)}\n\n"
            if once or await request.is_disconnected():
                return
            await asyncio.sleep(interval)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
