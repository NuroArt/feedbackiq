"""Application factory: middleware, error handlers, lifespan, static UI."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.errors import FeedbackIQError
from app.logging_config import configure_logging
from app.models import ErrorResponse
from app.observability import ObservabilityMiddleware
from app.routers.api import router
from app.services.pipeline import FeedbackPipeline

log = structlog.get_logger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings)
    # One pipeline for the process: the analyzer holds a pooled HTTP client, so
    # rebuilding it per request would mean a new TLS handshake per request.
    app.state.pipeline = FeedbackPipeline(settings)
    log.info("startup", **settings.redacted())
    yield
    log.info("shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings)

    app = FastAPI(
        title="FeedbackIQ",
        version="1.0.0",
        description=(
            "LLM-assisted customer feedback triage. Classifies raw feedback, "
            "clusters it into themes, files GitHub issues for high-severity "
            "themes, and posts a digest to Slack."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(ObservabilityMiddleware)
    app.include_router(router)

    def _request_id(request: Request) -> str | None:
        bound = structlog.contextvars.get_contextvars().get("request_id")
        return request.headers.get("X-Request-ID") or bound  # type: ignore[return-value]

    # ---- one error contract for every failure path ----------------------
    @app.exception_handler(FeedbackIQError)
    async def _domain_error(request: Request, exc: FeedbackIQError) -> JSONResponse:
        log.warning("domain_error", code=exc.code, message=exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=exc.code,
                message=exc.message,
                request_id=_request_id(request),
                detail=exc.detail,
            ).model_dump(mode="json"),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="validation_error",
                message="Request payload failed validation.",
                request_id=_request_id(request),
                detail=exc.errors()[:10],
            ).model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Log the stack trace; return nothing internal to the caller.
        log.error("unhandled_exception", error=str(exc), exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_error",
                message="An unexpected error occurred.",
                request_id=_request_id(request),
            ).model_dump(mode="json"),
        )

    # ---- minimal operator UI --------------------------------------------
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
