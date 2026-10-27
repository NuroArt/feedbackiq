"""Structured logging.

Every log line is a JSON object in non-local environments, so a log shipper
(CloudWatch -> OpenSearch, Datadog, Loki) can index fields without regex.
`request_id` is bound to a contextvar by the middleware, which means every log
emitted anywhere in the call stack carries it with no plumbing.
"""

from __future__ import annotations

import logging
import sys

import structlog
from structlog.typing import Processor

from app.config import Settings


def configure_logging(settings: Settings) -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
        force=True,
    )
    # Uvicorn's own loggers propagate into the root handler so access logs are
    # structured too, instead of appearing as unparsed text.
    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access", "httpx"):
        logging.getLogger(noisy).handlers.clear()
        logging.getLogger(noisy).propagate = True
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer = (
        structlog.processors.JSONRenderer()
        if settings.log_format == "json"
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
