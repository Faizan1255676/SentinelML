"""
FastAPI Request Correlation and Latency Timing Middleware.

Injects unique request correlation IDs (X-Request-ID), computes high-precision
request latency (X-Process-Time-Ms), and logs structured access metrics.
"""

import time
from uuid import uuid4
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from src.core.logger import get_logger

logger = get_logger("sentinelml.api.access")


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Captures request latency and correlates requests with unique IDs."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Extract or generate correlation ID
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        request.state.request_id = request_id

        start_time = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 3)
            logger.error(
                f"Unhandled exception on {request.method} {request.url.path}: {exc}",
                extra={
                    "request_id": request_id,
                    "latency_ms": duration_ms,
                    "status_code": 500,
                },
                exc_info=True,
            )
            raise exc

        duration_ms = round((time.perf_counter() - start_time) * 1000, 3)

        # Inject observability headers into response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.3f}"

        # Structured access log for non-health endpoints (or debug)
        if request.url.path != "/health":
            logger.info(
                f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms:.2f}ms)",
                extra={
                    "request_id": request_id,
                    "latency_ms": duration_ms,
                    "status_code": response.status_code,
                },
            )

        return response
