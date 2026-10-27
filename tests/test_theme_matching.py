"""Cross-run theme matching.

Every case here comes from the real double-run on 2026-10-25, where the model
re-worded four of five themes between two runs eight minutes apart and the
tracker filled with duplicates.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from app.config import Settings
from app.models import Category, Theme
from app.services.github_client import (
    CATEGORY_PREFIX,
    THEME_PREFIX,
    GitHubClient,
    _extract,
    fingerprint,
    render_issue_body,
)
from app.services.theme_matching import (
    MATCH_THRESHOLD,
    best_match,
    canonical_key,
    normalize_tokens,
    similarity,
)

# The actual pairs observed. Run 1 name, run 2 name.
OBSERVED_REWORDINGS = [
    ("Export fails with server error", "Report export fails with error"),
    ("CSV upload crashes losing data", "CSV upload crashes and loses data"),
    ("Slow dashboard and chart loading", "Slow dashboard and chart loading"),
    ("Unannounced renewal price increase", "Unannounced renewal price increase"),
]

MUST_NOT_MATCH = [
    ("Slow dashboard and chart loading", "CSV upload crashes losing data"),
    ("Billing settings hard to find", "Salesforce integration request"),
    ("Export fails with server error", "Slow dashboard and chart loading"),
    ("Unannounced renewal price increase", "Support response times too slow"),
]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def test_stemming_collapses_the_observed_variation():
    """losing/loses, crashes/crashing -- the drift that caused the duplicates."""
    assert normalize_tokens("CSV upload crashes losing data") == normalize_tokens(
        "CSV upload crashes and loses data"
    )


def test_word_order_does_not_change_the_key():
    assert canonical_key("bug", "export fails with server error") == canonical_key(
        "bug", "server error when export fails"
    )


def test_category_is_part_of_the_key():
    """The same words under a different category are a different problem."""
    assert canonical_key("bug", "slow export") != canonical_key("performance", "slow export")


def test_discriminating_words_are_not_stripped():
    """'error' and 'fails' separate a bug from a feature request; keep them."""
    tokens = normalize_tokens("Export fails with server error")
    assert "error" in tokens and "fail" in tokens
    assert "with" not in tokens  # but stopwords go


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("run1", "run2"), OBSERVED_REWORDINGS)
def test_every_observed_rewording_matches(run1, run2):
    score = similarity(run1, run2)
    assert score >= MATCH_THRESHOLD, f"{run1!r} vs {run2!r} scored {score:.2f}"


@pytest.mark.parametrize(("a", "b"), MUST_NOT_MATCH)
def test_distinct_themes_stay_distinct(a, b):
    score = similarity(a, b)
    assert score < MATCH_THRESHOLD, f"{a!r} vs {b!r} scored {score:.2f}"


def test_threshold_sits_in_the_gap_between_the_two_populations():
    """A threshold is only safe if the populations actually separate."""
    matches = [similarity(a, b) for a, b in OBSERVED_REWORDINGS]
    non = [similarity(a, b) for a, b in MUST_NOT_MATCH]
    assert min(matches) > max(non), (
        f"populations overlap: worst match {min(matches):.2f} vs best non-match {max(non):.2f}"
    )
    assert max(non) < MATCH_THRESHOLD <= min(matches)


def test_empty_and_junk_names_do_not_match_everything():
    assert similarity("", "anything") == 0.0
    assert similarity("the and of", "the and of") == 0.0  # all stopwords


# ---------------------------------------------------------------------------
# best_match
# ---------------------------------------------------------------------------
def test_best_match_requires_the_same_category():
    candidates = [("1", "performance", "Export fails with server error")]
    assert best_match("bug", "Report export fails with error", candidates) is None


def test_best_match_picks_the_highest_scoring_candidate():
    candidates = [
        ("1", "bug", "Export fails sometimes"),
        ("2", "bug", "Report export fails with error"),
    ]
    ref, score = best_match("bug", "Export fails with server error", candidates)
    assert ref == "2" and score > 0.5


# ---------------------------------------------------------------------------
# Issue body round-trip
# ---------------------------------------------------------------------------
def _theme(name="Export fails with server error", category=Category.BUG, sev=4):
    return Theme(
        name=name,
        category=category,
        item_count=2,
        max_severity=sev,
        avg_severity=4.0,
        negative_share=1.0,
        example_quotes=["q"],
        item_ids=["a", "b"],
    )


def test_issue_body_carries_the_theme_and_category_for_later_matching():
    body = render_issue_body(_theme(), "run_1")
    assert _extract(body, THEME_PREFIX) == "Export fails with server error"
    assert _extract(body, CATEGORY_PREFIX) == "bug"
    assert fingerprint(_theme()) in body


def test_extract_returns_none_for_a_body_without_markers():
    assert _extract("just a normal issue someone wrote", THEME_PREFIX) is None


# ---------------------------------------------------------------------------
# Client behaviour
# ---------------------------------------------------------------------------
def _settings():
    return Settings(github_token="t", github_repo="acme/app", llm_offline_mode=True)


def _issue(number: int, name: str, category: str = "bug") -> dict:
    return {
        "number": number,
        "html_url": f"https://github.com/acme/app/issues/{number}",
        "body": f"<!-- {CATEGORY_PREFIX}{category} -->\n<!-- {THEME_PREFIX}{name} -->",
    }


@respx.mock
async def test_candidates_come_from_a_label_listing_not_search():
    """Search is eventually consistent; a label listing is not."""
    listing = respx.get("https://api.github.com/repos/acme/app/issues").mock(
        return_value=httpx.Response(200, json=[_issue(1, "Export fails with server error")])
    )
    search = respx.get("https://api.github.com/search/issues")

    async with GitHubClient(_settings()) as gh:
        candidates = await gh.list_candidates()

    assert listing.called and not search.called
    assert candidates == [("1", "bug", "Export fails with server error")]


@respx.mock
async def test_reworded_theme_comments_instead_of_duplicating():
    """The exact failure from the live run: run 2 re-worded, filed a duplicate."""
    respx.get("https://api.github.com/repos/acme/app/issues").mock(
        return_value=httpx.Response(200, json=[_issue(1, "Export fails with server error")])
    )
    comment = respx.post("https://api.github.com/repos/acme/app/issues/1/comments").mock(
        return_value=httpx.Response(201, json={})
    )
    create = respx.post("https://api.github.com/repos/acme/app/issues")

    async with GitHubClient(_settings()) as gh:
        action = await gh.upsert_theme_issue(_theme("Report export fails with error"), "run_2")

    assert comment.called, "a re-worded theme must comment on the existing issue"
    assert not create.called, "and must not file a new one"
    assert action.action == "comment_on_issue"
    assert "similarity" in action.detail


@respx.mock
async def test_genuinely_new_theme_still_files_an_issue():
    """Matching must not be so loose that nothing new ever gets filed."""
    respx.get("https://api.github.com/repos/acme/app/issues").mock(
        return_value=httpx.Response(200, json=[_issue(1, "Export fails with server error")])
    )
    create = respx.post("https://api.github.com/repos/acme/app/issues").mock(
        return_value=httpx.Response(
            201, json={"number": 2, "html_url": "https://github.com/acme/app/issues/2"}
        )
    )

    async with GitHubClient(_settings()) as gh:
        action = await gh.upsert_theme_issue(
            _theme("Salesforce integration request", Category.FEATURE_REQUEST), "run_2"
        )

    assert create.called and action.action == "create_issue"


@respx.mock
async def test_two_similar_themes_in_one_run_do_not_both_file():
    """The new issue must be visible to the next theme in the same run."""
    respx.get("https://api.github.com/repos/acme/app/issues").mock(
        return_value=httpx.Response(200, json=[])
    )
    create = respx.post("https://api.github.com/repos/acme/app/issues").mock(
        return_value=httpx.Response(
            201, json={"number": 9, "html_url": "https://github.com/acme/app/issues/9"}
        )
    )
    comment = respx.post("https://api.github.com/repos/acme/app/issues/9/comments").mock(
        return_value=httpx.Response(201, json={})
    )

    async with GitHubClient(_settings()) as gh:
        shared: list[tuple[str, str, str]] = await gh.list_candidates()
        first = await gh.upsert_theme_issue(
            _theme("Export fails with server error"), "run_1", shared
        )
        second = await gh.upsert_theme_issue(
            _theme("Report export fails with error"), "run_1", shared
        )

    assert first.action == "create_issue"
    assert second.action == "comment_on_issue"
    assert create.call_count == 1 and comment.called


@respx.mock
async def test_pull_requests_are_ignored_in_the_listing():
    """GET /issues returns PRs too; a PR is not a feedback theme."""
    respx.get("https://api.github.com/repos/acme/app/issues").mock(
        return_value=httpx.Response(
            200,
            json=[
                _issue(1, "Export fails with server error"),
                {
                    "number": 2,
                    "pull_request": {"url": "x"},
                    "body": f"<!-- {CATEGORY_PREFIX}bug -->\n<!-- {THEME_PREFIX}a pr -->",
                },
            ],
        )
    )
    async with GitHubClient(_settings()) as gh:
        assert len(await gh.list_candidates()) == 1


# ---------------------------------------------------------------------------
# Rate limiting -- the second bug from the live run
# ---------------------------------------------------------------------------
@respx.mock
async def test_secondary_rate_limit_is_retried():
    """GitHub signals a write burst limit with 403 + a rate-limit message."""
    route = respx.get("https://api.github.com/repos/acme/app/issues").mock(
        side_effect=[
            httpx.Response(
                403,
                headers={"retry-after": "0"},
                json={"message": "You have exceeded a secondary rate limit"},
            ),
            httpx.Response(200, json=[]),
        ]
    )
    async with GitHubClient(_settings()) as gh:
        assert await gh.list_candidates() == []
    assert route.call_count == 2


@respx.mock
async def test_permission_403_is_not_retried_and_names_the_cause():
    """A 403 for permissions returns the same answer forever; fail fast."""
    route = respx.get("https://api.github.com/repos/acme/app/issues").mock(
        return_value=httpx.Response(
            403, json={"message": "Resource not accessible by personal access token"}
        )
    )
    from app.errors import IntegrationError

    async with GitHubClient(_settings()) as gh:
        with pytest.raises(IntegrationError) as caught:
            await gh.list_candidates()

    assert route.call_count == 1
    assert "not accessible" in caught.value.message


# ---------------------------------------------------------------------------
# Label independence
#
# From the live run: GitHub silently drops labels when the token lacks push
# access. Every issue came back with `"labels": []`, so a dedup lookup
# filtered on labels returned nothing and every run filed duplicates -- with
# no error anywhere to explain it.
# ---------------------------------------------------------------------------
def _unlabelled_issue(number: int, name: str, category: str = "bug") -> dict:
    """An issue as GitHub returns it when labels were dropped."""
    return {
        "number": number,
        "html_url": f"https://github.com/acme/app/issues/{number}",
        "labels": [],
        "body": f"<!-- {CATEGORY_PREFIX}{category} -->\n<!-- {THEME_PREFIX}{name} -->",
    }


@respx.mock
async def test_dedup_works_when_github_drops_every_label():
    """The regression. Identity lives in the body, not in metadata."""
    respx.get("https://api.github.com/repos/acme/app/issues").mock(
        return_value=httpx.Response(
            200,
            json=[
                _unlabelled_issue(1, "Export fails with server error"),
                _unlabelled_issue(2, "Slow dashboard and chart loading", "performance"),
            ],
        )
    )
    async with GitHubClient(_settings()) as gh:
        candidates = await gh.list_candidates()

    assert len(candidates) == 2, "label-less issues must still be found"
    assert ("1", "bug", "Export fails with server error") in candidates


@respx.mock
async def test_listing_does_not_filter_by_label():
    """A labels= filter would return nothing on a push-access-less token."""
    route = respx.get("https://api.github.com/repos/acme/app/issues").mock(
        return_value=httpx.Response(200, json=[])
    )
    async with GitHubClient(_settings()) as gh:
        await gh.list_candidates()

    query = str(route.calls[0].request.url)
    assert "labels=" not in query, f"listing must not depend on labels: {query}"
    assert "state=all" in query


@respx.mock
async def test_identical_theme_name_matches_an_unlabelled_issue():
    """Run 2 filed exact-name duplicates because candidates came back empty."""
    respx.get("https://api.github.com/repos/acme/app/issues").mock(
        return_value=httpx.Response(
            200, json=[_unlabelled_issue(1, "Slow dashboard and chart loading", "performance")]
        )
    )
    comment = respx.post("https://api.github.com/repos/acme/app/issues/1/comments").mock(
        return_value=httpx.Response(201, json={})
    )
    create = respx.post("https://api.github.com/repos/acme/app/issues")

    async with GitHubClient(_settings()) as gh:
        action = await gh.upsert_theme_issue(
            _theme("Slow dashboard and chart loading", Category.PERFORMANCE), "run_2"
        )

    assert comment.called and not create.called
    assert action.status == "created"


@respx.mock
async def test_failed_comment_does_not_look_like_broken_dedup():
    """Matching succeeded and no duplicate was filed; say exactly that."""
    respx.get("https://api.github.com/repos/acme/app/issues").mock(
        return_value=httpx.Response(
            200, json=[_unlabelled_issue(1, "Export fails with server error")]
        )
    )
    respx.post("https://api.github.com/repos/acme/app/issues/1/comments").mock(
        return_value=httpx.Response(403, json={"message": "Resource not accessible"})
    )
    create = respx.post("https://api.github.com/repos/acme/app/issues")

    async with GitHubClient(_settings()) as gh:
        action = await gh.upsert_theme_issue(_theme("Report export fails with error"), "run_2")

    assert not create.called, "a failed comment must never fall back to creating"
    assert action.action == "comment_on_issue"
    assert action.status == "failed"
    assert "no duplicate was filed" in action.detail
