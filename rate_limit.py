"""Lightweight per-IP POST rate limits (in-memory). Disable with RATE_LIMIT_ENABLED=0."""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

_lock = threading.Lock()
_events: dict[tuple[str, str], list[float]] = defaultdict(list)


def _client_ip(request: Request) -> str:
    xf = request.headers.get("x-forwarded-for")
    if xf:
        return xf.split(",")[0].strip()[:45] or "unknown"
    if request.client:
        return request.client.host or "unknown"
    return "unknown"


def _allow(bucket: str, ip: str, limit: int, window_sec: float) -> bool:
    if limit <= 0:
        return True
    key = (bucket, ip)
    now = time.monotonic()
    cutoff = now - window_sec
    with _lock:
        arr = _events[key]
        while arr and arr[0] < cutoff:
            arr.pop(0)
        if len(arr) >= limit:
            return False
        arr.append(now)
        return True


def rate_limit_enabled() -> bool:
    return os.environ.get("RATE_LIMIT_ENABLED", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        if not rate_limit_enabled():
            return await call_next(request)
        if request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path
        if path in ("/health", "/", "/docs", "/openapi.json", "/redoc") or path.startswith("/legal/"):
            return await call_next(request)

        ip = _client_ip(request)

        if path == "/billing/webhook":
            lim = int(os.environ.get("RATE_LIMIT_WEBHOOK_PER_MIN", "120") or "120")
            if not _allow("webhook", ip, lim, 60.0):
                return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)
            return await call_next(request)

        if request.method != "POST":
            return await call_next(request)

        if path == "/jobs" or path.startswith("/jobs/"):
            lim = int(os.environ.get("RATE_LIMIT_JOBS_PER_HOUR", "30") or "30")
            if not _allow("jobs", ip, lim, 3600.0):
                return JSONResponse(
                    {"detail": "Too many jobs from this network; try again later."},
                    status_code=429,
                )
        elif path.startswith("/auth/"):
            lim = int(os.environ.get("RATE_LIMIT_AUTH_PER_MIN", "25") or "25")
            if not _allow("auth", ip, lim, 60.0):
                return JSONResponse(
                    {"detail": "Too many auth requests; try again in a minute."},
                    status_code=429,
                )
        elif path.startswith("/billing/"):
            lim = int(os.environ.get("RATE_LIMIT_BILLING_PER_MIN", "60") or "60")
            if not _allow("billing", ip, lim, 60.0):
                return JSONResponse(
                    {"detail": "Too many billing requests; try again shortly."},
                    status_code=429,
                )
        elif path.startswith("/test/"):
            lim = int(os.environ.get("RATE_LIMIT_TEST_PER_HOUR", "12") or "12")
            if not _allow("test", ip, lim, 3600.0):
                return JSONResponse({"detail": "Test endpoint rate limit exceeded"}, status_code=429)

        return await call_next(request)
