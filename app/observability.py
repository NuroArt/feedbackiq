"""Observability: metrics, request correlation, and timing helpers.

Three signals are exposed:
  * logs    - structured JSON, correlated by request_id (logging_config.py)
  * metrics - Prometheus text format at GET /metrics
  * traces  - approximated here by per-stage timing on the pipeline run record.
              Swap `stage_timer` for an OpenTelemetry span to get real traces
              without touching call sites (see ARCHITECTURE.md).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import structlog
from prometheus_client import Counter, Gauge, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

log = structlog.get_logger(__name__)

# --------------------------------------------------------------------------
# Metric definitions. Label cardinality is deliberately low: `path` uses the
# route template (/v1/feedback/analyze), never the raw URL, so unbounded path
# params can't blow up the time series count.
# --------------------------------------------------------------------------
HTTP_REQUESTS = Counter(
    "feedbackiq_http_requests_total",
    "HTTP requests handled.",
    ["method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "feedbackiq_http_request_duration_seconds",
    "HTTP request latency.",
    ["method", "path"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
LLM_CALLS = Counter(
    "feedbackiq_llm_calls_total",
    "LLM API calls by outcome.",
    ["model", "outcome"],
)
LLM_LATENCY = Histogram(
    "feedbackiq_llm_call_duration_seconds",
    "LLM call latency.",
    ["model"],
    buckets=(0.5, 1, 2, 5, 10, 20, 40, 60, 120),
)
LLM_TOKENS = Counter(
    "feedbackiq_llm_tokens_total",
    "Tokens consumed, by direction. Multiply by unit price for spend.",
    ["model", "direction"],
)
INTEGRATION_CALLS = Counter(
    "feedbackiq_integration_calls_total",
    "Outbound integration calls by provider and outcome.",
    ["provider", "outcome"],
)
PIPELINE_RUNS = Counter(
    "feedbackiq_pipeline_runs_total",
    "Pipeline executions by outcome.",
    ["outcome"],
)
PIPELINE_STAGE_LATENCY = Histogram(
    "feedbackiq_pipeline_stage_duration_seconds",
    "Per-stage pipeline latency.",
    ["stage"],
    buckets=(0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
)
PIPELINE_IN_FLIGHT = Gauge(
    "feedbackiq_pipeline_in_flight",
    "Pipeline runs currently executing.",
)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Binds a request id, times the request, emits the access log + metrics."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        started = time.perf_counter()
        status = 500
        response: Response | None = None
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            elapsed = time.perf_counter() - started
            # Route template, not request.url.path -> bounded label cardinality.
            route = request.scope.get("route")
            path = getattr(route, "path", request.url.path)
            HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
            HTTP_LATENCY.labels(request.method, path).observe(elapsed)
            log.info(
                "http_request",
                method=request.method,
                path=path,
                status=status,
                duration_ms=round(elapsed * 1000, 2),
            )
            structlog.contextvars.clear_contextvars()


@contextmanager
def stage_timer(stage: str, **fields: object) -> Iterator[dict[str, object]]:
    """Time a pipeline stage, record the histogram, log start/finish.

    Yields a mutable dict so the body can attach result counts that end up in
    the completion log line.
    """
    extra: dict[str, object] = {}
    started = time.perf_counter()
    log.debug("stage_start", stage=stage, **fields)
    try:
        yield extra
    except Exception as exc:
        PIPELINE_STAGE_LATENCY.labels(stage).observe(time.perf_counter() - started)
        log.error("stage_failed", stage=stage, error=str(exc), exc_info=True, **fields)
        raise
    else:
        elapsed = time.perf_counter() - started
        PIPELINE_STAGE_LATENCY.labels(stage).observe(elapsed)
        log.info(
            "stage_complete",
            stage=stage,
            duration_ms=round(elapsed * 1000, 2),
            **fields,
            **extra,
        )


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:12]}"
