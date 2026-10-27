"""Pipeline orchestration -- the business workflow itself.

    ingest -> analyze (LLM) -> aggregate -> prioritize -> route -> notify

Two properties the design is built around:

* **Degrade, don't fail.** The LLM stage is the only one allowed to abort the
  run. GitHub or Slack failing is recorded as a failed `ActionRecord` and the
  caller still gets the analysis. An integration outage must not destroy work
  that already cost money to produce.
* **Deterministic ranking and summary.** Clustering, scoring, and the executive
  summary are computed from the structured analyses, not generated. Numbers in
  the Slack digest are therefore arithmetic, not model output -- nothing in the
  digest can be hallucinated.
"""

from __future__ import annotations

import time
from collections import defaultdict

import structlog

from app.config import Settings
from app.errors import IntegrationError
from app.models import (
    ActionRecord,
    AnalyzeRequest,
    AnalyzeResponse,
    FeedbackItem,
    ItemAnalysis,
    RunMetrics,
    Sentiment,
    StageTiming,
    Theme,
)
from app.observability import (
    PIPELINE_IN_FLIGHT,
    PIPELINE_RUNS,
    new_run_id,
    stage_timer,
)
from app.services.github_client import GitHubClient
from app.services.llm import StubAnalyzer, build_analyzer
from app.services.slack_client import SlackClient

log = structlog.get_logger(__name__)


def aggregate_themes(items: list[FeedbackItem], analyses: list[ItemAnalysis]) -> list[Theme]:
    """Cluster per-item analyses into themes, ranked by priority score."""
    by_id = {i.id: i for i in items}
    buckets: dict[str, list[ItemAnalysis]] = defaultdict(list)
    for a in analyses:
        buckets[a.theme.strip().lower()].append(a)

    themes: list[Theme] = []
    for group in buckets.values():
        severities = [a.severity for a in group]
        negatives = sum(1 for a in group if a.sentiment is Sentiment.NEGATIVE)
        enterprise = sum(
            1 for a in group if (src := by_id.get(a.id)) and src.customer_tier == "enterprise"
        )
        # Quote the highest-severity items -- those carry the most signal.
        ranked = sorted(group, key=lambda a: a.severity, reverse=True)
        quotes = [(by_id[a.id].text[:200] if a.id in by_id else a.summary) for a in ranked[:3]]
        themes.append(
            Theme(
                name=group[0].theme.strip(),
                category=group[0].category,
                item_count=len(group),
                max_severity=max(severities),
                avg_severity=round(sum(severities) / len(severities), 2),
                negative_share=round(negatives / len(group), 3),
                enterprise_count=enterprise,
                example_quotes=quotes,
                item_ids=[a.id for a in group],
            )
        )
    return sorted(themes, key=lambda t: t.priority_score, reverse=True)


def build_executive_summary(themes: list[Theme], metrics: RunMetrics) -> str:
    if not themes:
        return "No themes were extracted from this batch."
    total = sum(t.item_count for t in themes)
    weighted_neg = sum(t.negative_share * t.item_count for t in themes) / max(total, 1)
    top = themes[0]
    lines = [
        f"Analyzed {metrics.items_analyzed} feedback items into {len(themes)} themes; "
        f"{weighted_neg:.0%} of items carried negative sentiment.",
        f"Top issue: *{top.name}* — {top.item_count} reports at severity "
        f"{top.max_severity}/5"
        + (
            f", including {top.enterprise_count} enterprise account(s)."
            if top.enterprise_count
            else "."
        ),
    ]
    urgent = [t for t in themes[1:4] if t.max_severity >= 4]
    if urgent:
        lines.append(
            "Also urgent: " + "; ".join(f"{t.name} ({t.item_count})" for t in urgent) + "."
        )
    if metrics.items_failed:
        lines.append(
            f"{metrics.items_failed} item(s) could not be analyzed and were excluded."
        )
    return "\n".join(lines)


class FeedbackPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.analyzer = build_analyzer(settings)

    async def run(self, request: AnalyzeRequest) -> AnalyzeResponse:
        run_id = new_run_id()
        structlog.contextvars.bind_contextvars(run_id=run_id)
        dry_run = request.dry_run or self.settings.dry_run
        started = time.perf_counter()
        stages: list[StageTiming] = []

        PIPELINE_IN_FLIGHT.inc()
        try:
            # --- 1. analyze -------------------------------------------------
            t0 = time.perf_counter()
            with stage_timer("analyze", items=len(request.items)) as extra:
                result = await self.analyzer.analyze(request.items, request.product_context)
                extra["analyzed"] = len(result.analyses)
                extra["failed"] = len(result.failed_ids)
            stages.append(
                StageTiming(stage="analyze", duration_ms=(time.perf_counter() - t0) * 1000)
            )

            # --- 2. aggregate ----------------------------------------------
            t0 = time.perf_counter()
            with stage_timer("aggregate") as extra:
                themes = aggregate_themes(request.items, result.analyses)
                extra["themes"] = len(themes)
            stages.append(
                StageTiming(stage="aggregate", duration_ms=(time.perf_counter() - t0) * 1000)
            )

            metrics = RunMetrics(
                items_in=len(request.items),
                items_analyzed=len(result.analyses),
                items_failed=len(result.failed_ids),
                themes=len(themes),
                llm_calls=result.calls,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                stages=stages,
            )
            summary = build_executive_summary(themes, metrics)

            # --- 3. route to GitHub ----------------------------------------
            actions: list[ActionRecord] = []
            t0 = time.perf_counter()
            with stage_timer("route_github") as extra:
                actions += await self._file_issues(themes, run_id, request, dry_run)
                extra["issues"] = sum(1 for a in actions if a.status == "created")
            stages.append(
                StageTiming(
                    stage="route_github", duration_ms=(time.perf_counter() - t0) * 1000
                )
            )

            # --- 4. notify Slack -------------------------------------------
            t0 = time.perf_counter()
            with stage_timer("notify_slack"):
                links = [a.target for a in actions if a.provider == "github" and a.target]
                actions.append(
                    await self._notify(
                        run_id, summary, themes, metrics, links, request, dry_run
                    )
                )
            stages.append(
                StageTiming(
                    stage="notify_slack", duration_ms=(time.perf_counter() - t0) * 1000
                )
            )

            metrics.total_duration_ms = round((time.perf_counter() - started) * 1000, 2)
            metrics.stages = stages
            PIPELINE_RUNS.labels("success").inc()
            log.info(
                "pipeline_complete",
                items=metrics.items_in,
                themes=metrics.themes,
                duration_ms=metrics.total_duration_ms,
            )
            return AnalyzeResponse(
                run_id=run_id,
                mode="offline_stub" if isinstance(self.analyzer, StubAnalyzer) else "live",
                executive_summary=summary,
                themes=themes,
                analyses=result.analyses,
                actions=actions,
                metrics=metrics,
            )
        except Exception:
            PIPELINE_RUNS.labels("failure").inc()
            raise
        finally:
            PIPELINE_IN_FLIGHT.dec()
            structlog.contextvars.unbind_contextvars("run_id")

    # ------------------------------------------------------------------
    async def _file_issues(
        self,
        themes: list[Theme],
        run_id: str,
        request: AnalyzeRequest,
        dry_run: bool,
    ) -> list[ActionRecord]:
        threshold = self.settings.github_severity_threshold
        candidates = [t for t in themes if t.max_severity >= threshold][:5]

        if not request.create_issues:
            return [
                ActionRecord(
                    provider="github",
                    action="create_issue",
                    status="skipped",
                    detail="Disabled on the request.",
                )
            ]
        if not self.settings.github_enabled:
            return [
                ActionRecord(
                    provider="github",
                    action="create_issue",
                    status="skipped",
                    detail="GITHUB_TOKEN / GITHUB_REPO not configured.",
                )
            ]
        if not candidates:
            return [
                ActionRecord(
                    provider="github",
                    action="create_issue",
                    status="skipped",
                    detail=f"No theme reached severity {threshold}.",
                )
            ]
        if dry_run:
            return [
                ActionRecord(
                    provider="github",
                    action="create_issue",
                    status="dry_run",
                    detail=f"Would file: {t.name} (severity {t.max_severity}).",
                )
                for t in candidates
            ]

        actions: list[ActionRecord] = []
        async with GitHubClient(self.settings) as gh:
            # Fetch existing issues once and pass the list down. Each upsert
            # appends anything it creates, so a second near-identical theme in
            # the same run matches the issue the first one just filed.
            try:
                existing = await gh.list_candidates()
            except IntegrationError as exc:
                log.error("github_candidates_failed", error=exc.message)
                return [
                    ActionRecord(
                        provider="github",
                        action="create_issue",
                        status="failed",
                        detail=exc.message,
                    )
                ]

            for theme in candidates:
                try:
                    actions.append(await gh.upsert_theme_issue(theme, run_id, existing))
                except IntegrationError as exc:
                    # Non-fatal: record and continue with the next theme.
                    log.error("github_action_failed", theme=theme.name, error=exc.message)
                    actions.append(
                        ActionRecord(
                            provider="github",
                            action="create_issue",
                            status="failed",
                            detail=exc.message,
                        )
                    )
        return actions

    async def _notify(
        self,
        run_id: str,
        summary: str,
        themes: list[Theme],
        metrics: RunMetrics,
        links: list[str],
        request: AnalyzeRequest,
        dry_run: bool,
    ) -> ActionRecord:
        if not request.notify_slack:
            return ActionRecord(
                provider="slack",
                action="post_digest",
                status="skipped",
                detail="Disabled on the request.",
            )
        if not self.settings.slack_enabled:
            return ActionRecord(
                provider="slack",
                action="post_digest",
                status="skipped",
                detail="SLACK_WEBHOOK_URL not configured.",
            )
        if dry_run:
            return ActionRecord(
                provider="slack",
                action="post_digest",
                status="dry_run",
                detail=f"Would post digest with {len(themes)} themes.",
            )
        try:
            async with SlackClient(self.settings) as slack:
                return await slack.post_digest(run_id, summary, themes, metrics, links)
        except IntegrationError as exc:
            log.error("slack_action_failed", error=exc.message)
            return ActionRecord(
                provider="slack", action="post_digest", status="failed", detail=exc.message
            )
