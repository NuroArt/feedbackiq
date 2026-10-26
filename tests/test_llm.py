"""LLM layer: schema handling and the offline stub's contract parity."""

import pytest
from pydantic import ValidationError

from app.models import FeedbackItem, ItemAnalysis
from app.services.llm import StubAnalyzer, _tool_schema, build_analyzer


def test_tool_schema_is_self_contained(settings):
    schema = _tool_schema()
    assert "$defs" not in str(schema), "enum refs must be inlined for the provider"
    props = schema["properties"]["analyses"]["items"]["properties"]
    assert set(props) >= {"id", "sentiment", "category", "severity", "summary", "theme"}
    assert props["severity"]["minimum"] == 1
    assert props["severity"]["maximum"] == 5


def test_build_analyzer_falls_back_to_stub_without_a_key(settings):
    assert isinstance(build_analyzer(settings), StubAnalyzer)


async def test_stub_returns_one_analysis_per_item(settings):
    items = [FeedbackItem(id=str(i), text=t) for i, t in enumerate(
        ["crash on save", "so slow", "love it", "please add webhooks"])]
    result = await StubAnalyzer(settings).analyze(items)
    assert [a.id for a in result.analyses] == [i.id for i in items]
    assert result.calls == 0  # nothing was spent


async def test_stub_assigns_low_severity_to_praise(settings):
    result = await StubAnalyzer(settings).analyze(
        [FeedbackItem(id="1", text="Love it, works great, thank you!")]
    )
    assert result.analyses[0].severity == 1
    assert result.analyses[0].sentiment == "positive"


def test_out_of_range_severity_is_rejected():
    """The guard that stops malformed model output reaching GitHub."""
    with pytest.raises(ValidationError):
        ItemAnalysis(id="1", sentiment="negative", category="bug", severity=9,
                     summary="x", theme="t")


def test_unknown_category_is_rejected():
    with pytest.raises(ValidationError):
        ItemAnalysis(id="1", sentiment="negative", category="made_up", severity=3,
                     summary="x", theme="t")
