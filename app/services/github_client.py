"""GitHub integration: turns a high-priority theme into a triaged issue.

Idempotency is the point. The workflow runs on a schedule over overlapping
windows of feedback, so the naive version files the same issue every morning.

Two things make that work, both learned from a real double-run:

**Matching is on meaning, not on an exact string.** The first implementation
hashed the theme name and searched for that hash. The model re-words themes
between runs ("Export fails with server error" / "Report export fails with
error"), so the hashes differed and the tracker filled with duplicates. Theme
names are now treated as evidence rather than identifiers -- see
`theme_matching`.

**Candidates come from a label listing, not the search API.** Search is
eventually consistent, so an issue created moments ago may not be findable,
which means a second run can duplicate it no matter how good the matching is.
`GET /issues?labels=...` is read-after-write consistent and removes that whole
class of failure -- and with it the need to wait before re-running.
"""

from __future__ import annotations

import asyncio

import httpx
import structlog

from app.config import Settings
from app.errors import IntegrationError
from app.models import ActionRecord, Theme
from app.observability import INTEGRATION_CALLS
from app.services.theme_matching import best_match, canonical_key

log = structlog.get_logger(__name__)

FINGERPRINT_PREFIX = "feedbackiq-fingerprint:"
THEME_PREFIX = "feedbackiq-theme:"
CATEGORY_PREFIX = "feedbackiq-category:"


def fingerprint(theme: Theme) -> str:
    """Order- and wording-tolerant key. See theme_matching.canonical_key."""
    return canonical_key(theme.category.value, theme.name)


def render_issue_body(theme: Theme, run_id: str) -> str:
    quotes = "\n".join(f"> {q}" for q in theme.example_quotes) or "> (none captured)"
    return f"""\
**Auto-filed by FeedbackIQ** from customer feedback analysis.

| Signal | Value |
| --- | --- |
| Reports | {theme.item_count} |
| Max severity | {theme.max_severity}/5 |
| Avg severity | {theme.avg_severity} |
| Negative share | {theme.negative_share:.0%} |
| Enterprise accounts affected | {theme.enterprise_count} |
| Priority score | {theme.priority_score} |

### Representative quotes
{quotes}

### Source items
`{", ".join(theme.item_ids[:25])}`

<!-- {FINGERPRINT_PREFIX}{fingerprint(theme)} run:{run_id} -->
<!-- {CATEGORY_PREFIX}{theme.category.value} -->
<!-- {THEME_PREFIX}{theme.name} -->
"""


def _extract(body: str, prefix: str) -> str | None:
    """Pull a value out of one of the HTML-comment markers in an issue body."""
    for line in (body or "").splitlines():
        line = line.strip()
        if prefix in line:
            after = line.split(prefix, 1)[1]
            return after.split("-->", 1)[0].strip().split(" run:")[0].strip()
    return None


def _api_error_text(resp: httpx.Response) -> str:
    """The provider's own message, not just the status code.

    GitHub returns 403 for both "you lack permission" and "you tripped a
    secondary rate limit", which call for completely different responses from
    an operator. Only the body distinguishes them.
    """
    try:
        payload = resp.json()
    except Exception:
        return resp.text[:200]
    if isinstance(payload, dict):
        parts = [str(payload.get("message", ""))]
        for err in payload.get("errors", [])[:2]:
            if isinstance(err, dict) and err.get("message"):
                parts.append(str(err["message"]))
        return " | ".join(p for p in parts if p)[:300]
    return str(payload)[:200]


class GitHubClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self._client = client
        self._owned = client is None

    async def __aenter__(self) -> GitHubClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.settings.github_api_base,
                timeout=20.0,
                headers={
                    "Authorization": f"Bearer {self.settings.github_token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                    "User-Agent": "feedbackiq/1.0",
                },
            )
        return self

    async def __aexit__(self, *_exc) -> None:
        if self._owned and self._client is not None:
            await self._client.aclose()

    async def _request(self, method: str, url: str, **kw) -> httpx.Response:
        """One request, with backoff for GitHub's rate limiting.

        Writes are rate limited more aggressively than reads, and bursting
        several issue creations plus comments in a few seconds is exactly the
        shape that trips the secondary limit. GitHub signals it with a 403 and
        either a `Retry-After` header or `x-ratelimit-remaining: 0`.
        """
        assert self._client is not None
        attempts = 3
        for attempt in range(1, attempts + 1):
            try:
                resp = await self._client.request(method, url, **kw)
            except httpx.HTTPError as exc:
                INTEGRATION_CALLS.labels("github", "network_error").inc()
                raise IntegrationError("github", f"GitHub request failed: {exc}") from exc

            if resp.status_code < 400:
                INTEGRATION_CALLS.labels("github", "success").inc()
                return resp

            detail = _api_error_text(resp)
            throttled = resp.status_code in (403, 429) and (
                "rate limit" in detail.lower()
                or "abuse" in detail.lower()
                or resp.headers.get("retry-after")
                or resp.headers.get("x-ratelimit-remaining") == "0"
            )
            if throttled and attempt < attempts:
                wait = float(resp.headers.get("retry-after", 2 ** attempt))
                INTEGRATION_CALLS.labels("github", "rate_limited").inc()
                log.warning("github_rate_limited", wait_seconds=wait,
                            attempt=attempt, detail=detail[:120])
                await asyncio.sleep(min(wait, 60))
                continue

            INTEGRATION_CALLS.labels("github", f"http_{resp.status_code}").inc()
            raise IntegrationError(
                "github",
                f"GitHub returned {resp.status_code} for {method} {url}: {detail}",
                detail={"status": resp.status_code, "api_message": detail},
            )

        raise IntegrationError("github", "Exhausted retries against GitHub.")

    async def list_candidates(self) -> list[tuple[str, str, str]]:
        """Existing FeedbackIQ issues as (number, category, theme name).

        Identified by the markers in the issue body, NOT by label.

        Labels are not load-bearing on purpose. GitHub silently drops labels
        and assignees when the token lacks push access to the repository: the
        POST still returns 201, the issue exists, and the labels are simply
        absent. A dedup lookup filtered on `labels=` then returns an empty list
        forever and every run files duplicates -- with no error anywhere to
        explain it. That exact failure cost a live debugging session.

        The body markers are written by the same request that creates the
        issue, so they cannot be dropped independently of it. Anything we must
        read back later belongs in content we control, not in metadata the
        provider may discard.

        A listing rather than a search, because search is eventually consistent
        and an issue created seconds ago may not be findable yet.
        """
        repo = self.settings.github_repo
        candidates: list[tuple[str, str, str]] = []
        for page in range(1, 6):  # 500 issues, well past this workload
            resp = await self._request(
                "GET",
                f"/repos/{repo}/issues",
                params={"state": "all", "per_page": 100, "page": page},
            )
            rows = resp.json()
            if not isinstance(rows, list) or not rows:
                break
            for issue in rows:
                if "pull_request" in issue:  # /issues returns PRs too
                    continue
                body = issue.get("body") or ""
                name = _extract(body, THEME_PREFIX)
                category = _extract(body, CATEGORY_PREFIX)
                if name and category:
                    candidates.append((str(issue["number"]), category, name))
            if len(rows) < 100:
                break
        log.info("github_candidates_loaded", count=len(candidates))
        return candidates

    async def upsert_theme_issue(
        self, theme: Theme, run_id: str, candidates: list[tuple[str, str, str]] | None = None
    ) -> ActionRecord:
        repo = self.settings.github_repo
        if candidates is None:
            candidates = await self.list_candidates()

        match = best_match(theme.category.value, theme.name, candidates)
        if match:
            number, score = match
            url = f"https://github.com/{repo}/issues/{number}"
            try:
                await self._request(
                    "POST",
                    f"/repos/{repo}/issues/{number}/comments",
                    json={
                        "body": (
                            f"Still active: **{theme.item_count}** new reports in run "
                            f"`{run_id}` (max severity {theme.max_severity}/5, "
                            f"priority {theme.priority_score}).\n\n"
                            f"<sub>Matched as *{theme.name}* "
                            f"(similarity {score:.2f}).</sub>"
                        )
                    },
                )
            except IntegrationError as exc:
                # The valuable outcome already happened: the theme was matched
                # to an existing issue, so no duplicate was filed. Failing to
                # append the comment is a lesser, separate problem and must not
                # be reported as if dedup broke.
                log.warning("github_comment_failed", issue=number,
                            theme=theme.name, error=exc.message)
                return ActionRecord(
                    provider="github",
                    action="comment_on_issue",
                    status="failed",
                    target=url,
                    detail=(f"Matched existing issue #{number} "
                            f"(similarity {score:.2f}) so no duplicate was filed, "
                            f"but the comment could not be posted: {exc.message}"),
                )

            log.info("github_issue_updated", issue=number, theme=theme.name,
                     similarity=round(score, 3))
            return ActionRecord(
                provider="github",
                action="comment_on_issue",
                status="created",
                target=url,
                detail=(f"Matched existing issue #{number} "
                        f"(similarity {score:.2f}); appended run {run_id}."),
            )

        resp = await self._request(
            "POST",
            f"/repos/{repo}/issues",
            json={
                "title": f"[Feedback] {theme.name} ({theme.item_count} reports)"[:250],
                "body": render_issue_body(theme, run_id),
                "labels": [
                    self.settings.github_issue_label,
                    f"category:{theme.category.value}",
                    f"severity:{theme.max_severity}",
                ],
            },
        )
        issue = resp.json()
        if not issue.get("labels"):
            # Not an error -- the markers carry identity -- but the operator
            # should know their triage labels are not landing, because nothing
            # else will tell them. GitHub returns 201 either way.
            log.warning(
                "github_labels_dropped",
                issue=issue["number"],
                reason=(
                    "GitHub discards labels when the token lacks push access. "
                    "Dedup is unaffected (it reads body markers), but issues "
                    "will not carry category/severity labels."
                ),
            )
        log.info("github_issue_created", issue=issue["number"], theme=theme.name)
        # Make the new issue visible to later themes in this same run, so two
        # near-identical themes in one batch do not both create issues.
        candidates.append((str(issue["number"]), theme.category.value, theme.name))
        return ActionRecord(
            provider="github",
            action="create_issue",
            status="created",
            target=issue["html_url"],
            detail=f"Filed #{issue['number']} for theme '{theme.name}'.",
        )
