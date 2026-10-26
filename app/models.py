"""Domain models.

These are the contract at three boundaries: the HTTP API, the LLM (the analysis
schema is what the model is instructed to emit, and what we validate its output
against), and the integrations. Validating LLM output against a Pydantic model
is the single most load-bearing reliability control in the app -- a model that
hallucinates a field shape fails loudly here instead of corrupting a GitHub
issue three stages later.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, computed_field, field_validator


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class Category(StrEnum):
    BUG = "bug"
    FEATURE_REQUEST = "feature_request"
    USABILITY = "usability"
    PERFORMANCE = "performance"
    PRICING = "pricing"
    SUPPORT = "support"
    DOCS = "docs"
    OTHER = "other"


# --------------------------------------------------------------------------
# Inbound
# --------------------------------------------------------------------------
class FeedbackItem(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=8000)
    source: str = Field(default="unknown", max_length=64)
    customer_tier: Literal["free", "pro", "enterprise", "unknown"] = "unknown"
    submitted_at: datetime | None = None

    @field_validator("text")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("feedback text cannot be blank")
        return v


class AnalyzeRequest(BaseModel):
    items: list[FeedbackItem] = Field(min_length=1, max_length=500)
    product_context: str = Field(
        default="",
        max_length=1000,
        description="One or two lines about the product; sharpens LLM labelling.",
    )
    create_issues: bool = True
    notify_slack: bool = True
    dry_run: bool = False


# --------------------------------------------------------------------------
# LLM output schema -- the model is told to produce exactly this
# --------------------------------------------------------------------------
class ItemAnalysis(BaseModel):
    id: str
    sentiment: Sentiment
    category: Category
    severity: int = Field(ge=1, le=5, description="1 = cosmetic, 5 = blocking")
    summary: str = Field(max_length=300)
    theme: str = Field(
        max_length=80,
        description="Short canonical phrase used to cluster related feedback.",
    )


class Theme(BaseModel):
    """An aggregated cluster. This is the unit that becomes a GitHub issue."""

    name: str
    category: Category
    item_count: int
    max_severity: int
    avg_severity: float
    negative_share: float = Field(ge=0.0, le=1.0)
    enterprise_count: int = 0
    example_quotes: list[str] = Field(default_factory=list, max_length=3)
    item_ids: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def priority_score(self) -> float:
        """Blend of volume, pain, and revenue weight. Deterministic, auditable.

        `computed_field` so the score is part of the serialized API contract --
        a client sorting by it sees the same number the pipeline ranked on.

        Kept out of the LLM on purpose: ranking is business logic that needs to
        be stable across runs and explainable to a PM.
        """
        return round(
            self.item_count * 1.0
            + self.max_severity * 2.0
            + self.enterprise_count * 1.5
            + self.negative_share * 3.0,
            2,
        )


# --------------------------------------------------------------------------
# Outbound / results
# --------------------------------------------------------------------------
class ActionRecord(BaseModel):
    provider: Literal["github", "slack"]
    action: str
    status: Literal["created", "skipped", "failed", "dry_run"]
    target: str | None = None
    detail: str | None = None


class StageTiming(BaseModel):
    stage: str
    duration_ms: float


class RunMetrics(BaseModel):
    items_in: int
    items_analyzed: int
    items_failed: int
    themes: int
    llm_calls: int
    input_tokens: int = 0
    output_tokens: int = 0
    total_duration_ms: float = 0.0
    stages: list[StageTiming] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    run_id: str
    started_at: datetime = Field(default_factory=_utcnow)
    mode: Literal["live", "offline_stub"] = "live"
    executive_summary: str
    themes: list[Theme]
    analyses: list[ItemAnalysis]
    actions: list[ActionRecord]
    metrics: RunMetrics


class ErrorResponse(BaseModel):
    """Single error shape for every failure path. Clients parse one thing."""

    error: str
    message: str
    request_id: str | None = None
    detail: object | None = None
