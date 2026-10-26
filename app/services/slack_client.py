"""Slack integration: posts the run digest to an incoming webhook.

Incoming webhooks are used rather than a bot token because this is one-way
notification -- no scopes to over-grant, no token refresh, no Slack app review.
If the workflow later needs to read threads or react to messages, swap this for
a bot token and the Web API; the call site is one method wide.
"""

from __future__ import annotations

import httpx
import structlog

from app.config import Settings
from app.errors import IntegrationError
from app.models import ActionRecord, RunMetrics, Theme
from app.observability import INTEGRATION_CALLS

log = structlog.get_logger(__name__)

_SEVERITY_ICON = {5: ":rotating_light:", 4: ":red_circle:", 3: ":large_orange_circle:"}


def build_blocks(
    run_id: str,
    summary: str,
    themes: list[Theme],
    metrics: RunMetrics,
    issue_links: list[str],
) -> list[dict]:
    blocks: list[dict] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "Customer feedback digest"},
        },
        {"type": "section", "text": {"type": "mrkdwn", "text": summary[:2900]}},
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"`{run_id}` · {metrics.items_analyzed} items · "
                        f"{metrics.themes} themes · "
                        f"{metrics.total_duration_ms:.0f}ms · "
                        f"{metrics.input_tokens + metrics.output_tokens} tokens"
                    ),
                }
            ],
        },
        {"type": "divider"},
    ]
    for theme in themes[:5]:
        icon = _SEVERITY_ICON.get(theme.max_severity, ":white_circle:")
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"{icon} *{theme.name}*\n"
                        f"{theme.item_count} reports · severity {theme.max_severity}/5 · "
                        f"{theme.negative_share:.0%} negative · "
                        f"priority *{theme.priority_score}*"
                    ),
                },
            }
        )
    if issue_links:
        links = " · ".join(f"<{u}|issue>" for u in issue_links[:5])
        blocks.append(
            {"type": "context", "elements": [{"type": "mrkdwn", "text": f"Filed: {links}"}]}
        )
    return blocks


class SlackClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self._client = client
        self._owned = client is None

    async def __aenter__(self) -> SlackClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=15.0)
        return self

    async def __aexit__(self, *_exc) -> None:
        if self._owned and self._client is not None:
            await self._client.aclose()

    async def post_digest(
        self,
        run_id: str,
        summary: str,
        themes: list[Theme],
        metrics: RunMetrics,
        issue_links: list[str],
    ) -> ActionRecord:
        assert self._client is not None
        payload = {
            "text": f"Customer feedback digest ({metrics.items_analyzed} items)",
            "blocks": build_blocks(run_id, summary, themes, metrics, issue_links),
        }
        try:
            resp = await self._client.post(
                str(self.settings.slack_webhook_url), json=payload
            )
        except httpx.HTTPError as exc:
            INTEGRATION_CALLS.labels("slack", "network_error").inc()
            raise IntegrationError("slack", f"Slack request failed: {exc}") from exc

        if resp.status_code >= 400:
            INTEGRATION_CALLS.labels("slack", f"http_{resp.status_code}").inc()
            raise IntegrationError(
                "slack",
                f"Slack webhook returned {resp.status_code}",
                detail=resp.text[:200],
            )
        INTEGRATION_CALLS.labels("slack", "success").inc()
        log.info("slack_digest_posted", run_id=run_id, themes=len(themes))
        return ActionRecord(
            provider="slack",
            action="post_digest",
            status="created",
            detail=f"Posted digest with {len(themes)} themes.",
        )
