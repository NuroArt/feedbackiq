"""HTTP routes."""

from __future__ import annotations

import csv
import io

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config import Settings, get_settings
from app.deps import require_api_key
from app.models import AnalyzeRequest, AnalyzeResponse, FeedbackItem
from app.services.pipeline import FeedbackPipeline

router = APIRouter()


def get_pipeline(request: Request) -> FeedbackPipeline:
    """Reuse the pipeline built at startup (keeps the HTTP/LLM clients warm)."""
    return request.app.state.pipeline


# --------------------------------------------------------------------------
# Operational endpoints -- unauthenticated by design so probes stay simple.
# --------------------------------------------------------------------------
@router.get("/healthz", tags=["ops"], summary="Liveness probe")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz", tags=["ops"], summary="Readiness probe")
async def readyz(settings: Settings = Depends(get_settings)) -> dict[str, object]:
    """Reports dependency wiring. Returns 200 even when integrations are off:
    the service is genuinely able to serve traffic in stub mode."""
    return {
        "status": "ready",
        "llm": "live" if settings.llm_enabled else "offline_stub",
        "github": settings.github_enabled,
        "slack": settings.slack_enabled,
        "environment": settings.environment,
    }


@router.get("/metrics", tags=["ops"], summary="Prometheus metrics")
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# --------------------------------------------------------------------------
# Workflow endpoints
# --------------------------------------------------------------------------
@router.post(
    "/v1/feedback/analyze",
    response_model=AnalyzeResponse,
    tags=["workflow"],
    summary="Run the feedback workflow over a batch of items",
    dependencies=[Depends(require_api_key)],
)
async def analyze(
    payload: AnalyzeRequest,
    pipeline: FeedbackPipeline = Depends(get_pipeline),
) -> AnalyzeResponse:
    return await pipeline.run(payload)


@router.post(
    "/v1/feedback/analyze-csv",
    response_model=AnalyzeResponse,
    tags=["workflow"],
    summary="Same workflow, from a CSV export",
    dependencies=[Depends(require_api_key)],
)
async def analyze_csv(
    file: UploadFile = File(..., description="Columns: id, text[, source, customer_tier]"),
    pipeline: FeedbackPipeline = Depends(get_pipeline),
) -> AnalyzeResponse:
    """Most teams' feedback lives in a Zendesk/Typeform CSV export, so the
    workflow accepts one directly rather than demanding an ETL job first."""
    raw = (await file.read()).decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(raw))
    items = [
        FeedbackItem(
            id=(row.get("id") or f"row-{n}").strip(),
            text=(row.get("text") or row.get("feedback") or "").strip(),
            source=(row.get("source") or file.filename or "csv")[:64],
            customer_tier=(row.get("customer_tier") or "unknown").strip().lower()  # type: ignore[arg-type]
            if (row.get("customer_tier") or "unknown").strip().lower()
            in {"free", "pro", "enterprise", "unknown"}
            else "unknown",
        )
        for n, row in enumerate(reader, start=1)
        if (row.get("text") or row.get("feedback") or "").strip()
    ]
    return await pipeline.run(AnalyzeRequest(items=items))
