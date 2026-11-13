import logging
import time
import uuid

from .logging import _ctx, log_event

log = logging.getLogger("access")
_QUIET_PATHS = {"/healthz", "/metrics"}  # scrapers and probes: DEBUG instead of INFO


class RequestLogMiddleware:
    """Pure ASGI (no BaseHTTPMiddleware overhead). Adds X-Request-ID and one `http.request` event per request."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request_id = uuid.uuid4().hex[:16]
        started = time.perf_counter()
        status = 500

        async def send_with_id(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"] = [*message.get("headers", []), (b"x-request-id", request_id.encode())]
            await send(message)

        token = _ctx.set({**_ctx.get(), "request_id": request_id})
        try:
            await self.app(scope, receive, send_with_id)
        finally:
            route = scope.get("route")
            path = getattr(route, "path", None) or scope["path"]  # template, e.g. /runs/{run_id}: low cardinality
            fields = {"method": scope["method"], "path": path, "status": status,
                      "duration_ms": round((time.perf_counter() - started) * 1000, 1)}
            run_id = (scope.get("path_params") or {}).get("run_id")
            if run_id:
                fields["run_id"] = run_id
            quiet = scope["path"] in _QUIET_PATHS or scope["path"].startswith("/dashboard")  # polled every second
            level = logging.DEBUG if quiet else logging.INFO
            log_event(log, "http.request", level=level, **fields)
            _ctx.reset(token)
