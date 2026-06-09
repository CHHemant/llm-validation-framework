"""API middleware for auth and rate limits."""

from __future__ import annotations

import os
import time
from collections import deque

from fastapi import HTTPException, Request, status
from starlette.responses import JSONResponse


class APIKeyAuthMiddleware:
    """Simple API key check via X-API-Key header if APP_API_KEY is configured."""

    def __init__(self, app):
        self.app = app
        self.expected_key = os.getenv("APP_API_KEY")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.expected_key:
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin1").lower(): v.decode("latin1") for k, v in scope.get("headers", [])}
        if headers.get("x-api-key") != self.expected_key:
            response = JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid API key"},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


class InMemoryRateLimiter:
    """Very lightweight per-client rate limiter for local/single-process usage."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = {}

    def check(self, request: Request) -> None:
        client_host = request.client.host if request.client else "unknown"
        now = time.time()

        bucket = self._hits.setdefault(client_host, deque())
        while bucket and now - bucket[0] > self.window_seconds:
            bucket.popleft()

        if len(bucket) >= self.max_requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {self.max_requests}/{self.window_seconds}s",
            )

        bucket.append(now)
