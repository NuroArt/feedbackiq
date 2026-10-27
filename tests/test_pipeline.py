"""Pipeline logic: clustering, ranking, and graceful degradation."""

import httpx
import pytest
import respx

from app.config import Settings
from app.errors import IntegrationError
from app.models import AnalyzeRequest, Category, ItemAnalysis, Sentiment, Theme
from app.services.github_client import CATEGORY_PREFIX, THEME_PREFIX, GitHubClient
from app.services.pipeline import FeedbackPipeline, aggregate_themes, build_executive_summary


def _analysis(id_, theme, sev, sentiment=Sentiment.NEGATIVE, cat=Category.BUG):
    return ItemAnalysis(
        id=id_, sentiment=sentiment, category=cat, severity=sev, summary="s", theme=theme
    )


def test_aggregate_clusters_by_theme_case_insensitively(items):
    analyses = [
        _analysis("1", "Export crashes", 5),
        _analysis("2", "export crashes", 4),
        _analysis("3", "Slow dashboard", 3, cat=Category.PERFORMANCE),
    ]
    themes = aggregate_themes(items, analyses)
    assert len(themes) == 2
    top = themes[0]
    assert top.item_count == 2
    assert top.max_severity == 5
    assert top.avg_severity == 4.5
    assert top.enterprise_count == 1  # item "1" is the enterprise tier item


def test_priority_score_ranks_volume_severity_and_revenue():
    low = Theme(
        name="a",
        category=Category.DOCS,
        item_count=1,
        max_severity=2,
        avg_severity=2,
        negative_share=0.0,
    )
    high = Theme(
        name="b",
        category=Category.BUG,
        item_count=6,
        max_severity=5,
        avg_severity=4.0,
        negative_share=1.0,
        enterprise_count=3,
    )
    assert high.priority_score > low.priority_score


def test_executive_summary_reports_real_numbers(items):
    analyses = [_analysis("1", "Export crashes", 5), _analysis("2", "Export crashes", 4)]
    themes = aggregate_themes(items, analyses)
    from app.models import RunMetrics

    metrics = RunMetrics(
        items_in=2, items_analyzed=2, items_failed=0, themes=len(themes), llm_calls=1
    )
    summary = build_executive_summary(themes, metrics)
    assert "Export crashes" in summary
    assert "2 reports" in summary


async def test_pipeline_skips_integrations_when_unconfigured(settings, items):
    out = await FeedbackPipeline(settings).run(AnalyzeRequest(items=items))
    statuses = {a.provider: a.status for a in out.actions}
    assert statuses == {"github": "skipped", "slack": "skipped"}
    assert out.metrics.items_analyzed == len(items)


async def test_dry_run_reports_intent_without_writing(items):
    s = Settings(
        llm_offline_mode=True,
        github_token="t",
        github_repo="acme/app",
        slack_webhook_url="https://hooks.slack.test/x",
    )
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(url__startswith="https://").mock(return_value=httpx.Response(200))
        out = await FeedbackPipeline(s).run(AnalyzeRequest(items=items, dry_run=True))
    assert not route.called, "dry_run must not make outbound calls"
    assert {a.status for a in out.actions} == {"dry_run"}


@respx.mock
async def test_github_failure_degrades_instead_of_aborting(items):
    s = Settings(llm_offline_mode=True, github_token="t", github_repo="acme/app")
    respx.get("https://api.github.com/repos/acme/app/issues").mock(
        return_value=httpx.Response(503, json={"message": "unavailable"})
    )
    out = await FeedbackPipeline(s).run(AnalyzeRequest(items=items))
    gh = [a for a in out.actions if a.provider == "github"]
    assert all(a.status == "failed" for a in gh)
    # The expensive work still came back to the caller.
    assert out.themes and out.analyses


@respx.mock
async def test_existing_issue_is_commented_not_duplicated():
    """Dedup now matches on meaning via a label listing, not a fingerprint search.

    Cross-run re-wording is covered in depth in tests/test_theme_matching.py.
    """
    s = Settings(github_token="t", github_repo="acme/app")
    theme = Theme(
        name="Export crashes",
        category=Category.BUG,
        item_count=3,
        max_severity=5,
        avg_severity=4.3,
        negative_share=1.0,
    )
    respx.get("https://api.github.com/repos/acme/app/issues").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "number": 42,
                    "html_url": "https://github.com/acme/app/issues/42",
                    "body": (
                        f"<!-- {CATEGORY_PREFIX}bug -->\n<!-- {THEME_PREFIX}Export crashes -->"
                    ),
                }
            ],
        )
    )
    comment = respx.post("https://api.github.com/repos/acme/app/issues/42/comments").mock(
        return_value=httpx.Response(201, json={})
    )
    create = respx.post("https://api.github.com/repos/acme/app/issues")

    async with GitHubClient(s) as gh:
        action = await gh.upsert_theme_issue(theme, "run_test")

    assert comment.called and not create.called
    assert action.action == "comment_on_issue"
    assert action.target.endswith("/42")


@respx.mock
async def test_github_client_raises_typed_error():
    s = Settings(github_token="t", github_repo="acme/app")
    respx.get("https://api.github.com/repos/acme/app/issues").mock(
        return_value=httpx.Response(401, json={"message": "Bad credentials"})
    )
    with pytest.raises(IntegrationError) as exc:
        async with GitHubClient(s) as gh:
            await gh.list_candidates()
    assert exc.value.provider == "github"
