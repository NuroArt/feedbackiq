"""Tests for the live LLM code path.

Every other test in this suite exercises `StubAnalyzer`. This file drives the
real `LLMAnalyzer` -- retry policy, tool-call parsing, usage metering, batch
splitting, partial-failure accounting -- against a fake Anthropic client.

Why this matters: the first real API call is the single riskiest moment in the
project, and everything it depends on is testable without a key. A fake client
cannot tell you the provider accepts your schema, but it can tell you that
every line of code around the call does what it claims.

The one thing these tests deliberately do NOT assert is that the generated
`input_schema` is valid to Anthropic. That requires the provider. See
`scripts/preflight.py`.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from anthropic import APIStatusError

from app.config import Settings
from app.errors import LLMError, LLMPermanentError
from app.models import FeedbackItem
from app.services.llm import LLMAnalyzer


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    """Strip the retry wait so tests run in milliseconds, not minutes.

    Patches the symbol as imported into the module, which keeps the production
    wait policy intact and visible in the source.
    """
    from tenacity import wait_none

    monkeypatch.setattr("app.services.llm.wait_exponential_jitter", lambda **_: wait_none())


def live_settings(**kw) -> Settings:
    return Settings(anthropic_api_key="sk-test-not-a-real-key", llm_offline_mode=False, **kw)


def items(n: int) -> list[FeedbackItem]:
    return [FeedbackItem(id=f"i{k}", text=f"feedback number {k}") for k in range(n)]


# ---------------------------------------------------------------------------
# Fake provider
# ---------------------------------------------------------------------------
def tool_use_message(ids: list[str], *, input_tokens=100, output_tokens=50, **overrides):
    """A message shaped like a successful forced tool call."""
    analyses = [
        {
            "id": i,
            "sentiment": "negative",
            "category": "bug",
            "severity": 4,
            "summary": f"summary for {i}",
            "theme": "Export crashes",
            **overrides,
        }
        for i in ids
    ]
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="ignored preamble"),
            SimpleNamespace(
                type="tool_use", name="record_analysis", input={"analyses": analyses}
            ),
        ],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def status_error(code: int) -> APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return APIStatusError(
        f"http {code}", response=httpx.Response(code, request=request), body=None
    )


class FakeMessages:
    """Replays a scripted sequence of responses; records every call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(step, BaseException):
            raise step
        if callable(step):
            return step(kwargs)
        return step


def analyzer_with(script, **settings_kw) -> tuple[LLMAnalyzer, FakeMessages]:
    analyzer = LLMAnalyzer(live_settings(**settings_kw))
    fake = FakeMessages(script)
    analyzer.client = SimpleNamespace(messages=fake)
    return analyzer, fake


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------
async def test_request_forces_the_tool_call():
    analyzer, fake = analyzer_with([tool_use_message(["i0"])])
    await analyzer.analyze(items(1), "product context here")

    call = fake.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "record_analysis"}
    assert call["tools"][0]["name"] == "record_analysis"
    assert call["model"] == analyzer.settings.llm_model
    assert "severity" in str(call["tools"][0]["input_schema"])


async def test_prompt_carries_item_ids_and_product_context():
    analyzer, fake = analyzer_with([tool_use_message(["i0", "i1"])])
    await analyzer.analyze(items(2), "B2B analytics SaaS")

    content = fake.calls[0]["messages"][0]["content"]
    assert "B2B analytics SaaS" in content
    assert '"id": "i0"' in content and '"id": "i1"' in content


# ---------------------------------------------------------------------------
# Response handling
# ---------------------------------------------------------------------------
async def test_tool_use_block_is_found_past_a_text_block():
    """The provider may emit text before the tool call; it must be skipped."""
    analyzer, _ = analyzer_with([tool_use_message(["i0"])])
    result = await analyzer.analyze(items(1))
    assert [a.id for a in result.analyses] == ["i0"]


async def test_usage_is_metered_from_the_response():
    msg = tool_use_message(["i0"], input_tokens=1234, output_tokens=567)
    analyzer, _ = analyzer_with([msg])
    result = await analyzer.analyze(items(1))
    assert result.input_tokens == 1234
    assert result.output_tokens == 567
    assert result.calls == 1


async def test_missing_tool_use_block_is_an_error():
    no_tool = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="I'd rather chat")],
        usage=SimpleNamespace(input_tokens=10, output_tokens=2),
    )
    analyzer, fake = analyzer_with([no_tool], llm_max_attempts=2)
    with pytest.raises(LLMError):
        await analyzer.analyze(items(1))
    assert len(fake.calls) == 2, "a missing tool call is transient; it should retry"


async def test_schema_invalid_output_is_rejected_not_passed_downstream():
    """The guard that stops severity=99 reaching a GitHub issue."""
    bad = tool_use_message(["i0"], severity=99)
    analyzer, _ = analyzer_with([bad], llm_max_attempts=1)
    with pytest.raises(LLMError):
        await analyzer.analyze(items(1))


# ---------------------------------------------------------------------------
# Retry policy -- the part that spends money when it's wrong
# ---------------------------------------------------------------------------
async def test_429_is_retried_then_succeeds():
    analyzer, fake = analyzer_with(
        [status_error(429), tool_use_message(["i0"])], llm_max_attempts=3
    )
    result = await analyzer.analyze(items(1))
    assert len(result.analyses) == 1
    assert len(fake.calls) == 2


async def test_500_is_retried():
    analyzer, fake = analyzer_with(
        [status_error(503), tool_use_message(["i0"])], llm_max_attempts=3
    )
    await analyzer.analyze(items(1))
    assert len(fake.calls) == 2


async def test_400_is_not_retried():
    """Regression test: a malformed request returns the same 400 every time.

    Retrying it spends the full backoff ladder to reach the identical answer
    and buries the real cause under a retry-exhausted message.
    """
    analyzer, fake = analyzer_with([status_error(400)], llm_max_attempts=3)
    with pytest.raises(LLMError):
        await analyzer.analyze(items(1))
    assert len(fake.calls) == 1, "a 400 must fail fast, not burn the retry budget"


async def test_401_is_not_retried():
    analyzer, fake = analyzer_with([status_error(401)], llm_max_attempts=3)
    with pytest.raises(LLMError):
        await analyzer.analyze(items(1))
    assert len(fake.calls) == 1


async def test_permanent_error_is_distinguishable_from_transient():
    """So an operator reading the log knows whether to fix config or wait."""
    analyzer, _ = analyzer_with([status_error(400)], llm_max_attempts=1)
    with pytest.raises(LLMError) as exc:
        await analyzer._analyze_batch(items(1), "")
    assert isinstance(exc.value, LLMPermanentError)
    assert exc.value.code == "llm_rejected"


async def test_retries_are_bounded_by_max_attempts():
    analyzer, fake = analyzer_with([status_error(503)], llm_max_attempts=3)
    with pytest.raises(LLMError):
        await analyzer.analyze(items(1))
    assert len(fake.calls) == 3


async def test_timeout_is_retried():
    analyzer, fake = analyzer_with(
        [TimeoutError(), tool_use_message(["i0"])], llm_max_attempts=3
    )
    await analyzer.analyze(items(1))
    assert len(fake.calls) == 2


# ---------------------------------------------------------------------------
# Batching and partial failure
# ---------------------------------------------------------------------------
async def test_items_are_split_into_batches():
    def respond(kwargs):
        content = kwargs["messages"][0]["content"]
        ids = [f"i{k}" for k in range(12) if f'"id": "i{k}"' in content]
        return tool_use_message(ids)

    analyzer, fake = analyzer_with([respond], llm_batch_size=5)
    result = await analyzer.analyze(items(12))
    assert len(fake.calls) == 3  # 5 + 5 + 2
    assert len(result.analyses) == 12
    assert result.calls == 3


async def test_one_failing_batch_does_not_lose_the_others():
    """The degradation property: a partial run still returns work."""
    state = {"n": 0}

    def respond(kwargs):
        state["n"] += 1
        if state["n"] == 1:  # first batch always fails
            raise status_error(500)
        content = kwargs["messages"][0]["content"]
        ids = [f"i{k}" for k in range(10) if f'"id": "i{k}"' in content]
        return tool_use_message(ids)

    analyzer, _ = analyzer_with([respond], llm_batch_size=5, llm_max_attempts=1)
    result = await analyzer.analyze(items(10))
    assert len(result.analyses) == 5
    assert len(result.failed_ids) == 5
    assert result.calls == 1


async def test_items_the_model_silently_omits_are_counted_as_failed():
    """A model that returns 3 of 5 must not look like a clean run."""
    analyzer, _ = analyzer_with([tool_use_message(["i0", "i1", "i2"])])
    result = await analyzer.analyze(items(5))
    assert len(result.analyses) == 3
    assert set(result.failed_ids) == {"i3", "i4"}


async def test_total_failure_raises_rather_than_returning_empty():
    analyzer, _ = analyzer_with([status_error(500)], llm_batch_size=5, llm_max_attempts=1)
    with pytest.raises(LLMError) as exc:
        await analyzer.analyze(items(10))
    assert "failed_ids" in str(exc.value.detail)


async def test_concurrency_is_bounded():
    """Unbounded fan-out on a large upload is how you get rate-limited."""
    live = {"now": 0, "peak": 0}

    async def respond(kwargs):
        live["now"] += 1
        live["peak"] = max(live["peak"], live["now"])
        await asyncio.sleep(0.01)
        live["now"] -= 1
        content = kwargs["messages"][0]["content"]
        return tool_use_message([f"i{k}" for k in range(40) if f'"id": "i{k}"' in content])

    analyzer = LLMAnalyzer(live_settings(llm_batch_size=2))

    class Async:
        async def create(self, **kwargs):
            return await respond(kwargs)

    analyzer.client = SimpleNamespace(messages=Async())
    await analyzer.analyze(items(40))
    assert live["peak"] <= 4, f"semaphore breached: {live['peak']} concurrent calls"


# ---------------------------------------------------------------------------
# Tool-choice capability negotiation
#
# Found during the first live run: claude-sonnet-5-5 rejects
# tool_choice={"type":"tool"} with a 400. Forcing is a per-model capability,
# so the client negotiates instead of hardcoding whichever mode works today.
# ---------------------------------------------------------------------------
def tool_choice_error() -> APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return APIStatusError(
        "bad request",
        response=httpx.Response(400, request=request),
        body={
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": 'tool_choice: type "tool" and "any" are not supported '
                "for this model.",
            },
        },
    )


async def test_forced_tool_choice_is_the_default():
    analyzer, fake = analyzer_with([tool_use_message(["i0"])])
    await analyzer.analyze(items(1))
    assert fake.calls[0]["tool_choice"] == {"type": "tool", "name": "record_analysis"}


async def test_auto_can_be_configured_up_front():
    analyzer, fake = analyzer_with([tool_use_message(["i0"])], llm_tool_choice="auto")
    await analyzer.analyze(items(1))
    assert fake.calls[0]["tool_choice"] == {"type": "auto"}


async def test_unsupported_forcing_downgrades_and_succeeds():
    analyzer, fake = analyzer_with(
        [tool_choice_error(), tool_use_message(["i0"])], llm_max_attempts=3
    )
    result = await analyzer.analyze(items(1))

    assert len(result.analyses) == 1, "the downgrade must recover, not just log"
    assert fake.calls[0]["tool_choice"]["type"] == "tool"
    assert fake.calls[1]["tool_choice"] == {"type": "auto"}


async def test_downgrade_persists_across_batches():
    """Otherwise every batch pays for the same rejected call."""
    state = {"n": 0}

    def respond(kwargs):
        state["n"] += 1
        if state["n"] == 1:
            raise tool_choice_error()
        content = kwargs["messages"][0]["content"]
        return tool_use_message([f"i{k}" for k in range(10) if f'"id": "i{k}"' in content])

    analyzer, fake = analyzer_with([respond], llm_batch_size=2, llm_max_attempts=3)
    result = await analyzer.analyze(items(10))

    assert len(result.analyses) == 10
    later = [c["tool_choice"] for c in fake.calls[1:]]
    assert all(tc == {"type": "auto"} for tc in later), later


async def test_other_400s_still_fail_fast_without_downgrading():
    """The downgrade must be narrow: only the tool_choice capability error."""
    analyzer, fake = analyzer_with([status_error(400)], llm_max_attempts=3)
    with pytest.raises(LLMPermanentError):
        await analyzer._analyze_batch(items(1), "")
    assert len(fake.calls) == 1
    assert analyzer._forced is True, "an unrelated 400 must not change the mode"


async def test_permanent_error_carries_the_api_message():
    """A status code alone sends an operator to the logs to find the cause."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    exc = APIStatusError(
        "bad",
        response=httpx.Response(404, request=request),
        body={
            "type": "error",
            "error": {"type": "not_found_error", "message": "model: nonexistent-model"},
        },
    )
    analyzer, _ = analyzer_with([exc], llm_max_attempts=1)
    with pytest.raises(LLMPermanentError) as caught:
        await analyzer._analyze_batch(items(1), "")
    assert "model: nonexistent-model" in caught.value.message
    assert caught.value.detail["status"] == 404


# ---------------------------------------------------------------------------
# Regressions from the first live eval run
# ---------------------------------------------------------------------------
async def test_concurrent_batches_all_survive_one_downgrade():
    """The bug that killed the first live eval.

    Batches run concurrently, so several are already in flight with _forced
    still true when the first 400 arrives. The downgrade must be idempotent: a
    sibling batch that hits the same 400 after the flag is cleared has to
    retry, not fall through to LLMPermanentError and take the whole run down.

    The sleep is load-bearing -- it creates the await point that lets all four
    batches enter the call while still in forced mode, which is exactly what
    happened against the real API.
    """
    seen: list[str] = []

    class RacingClient:
        async def create(self, **kwargs):
            mode = kwargs["tool_choice"]["type"]
            seen.append(mode)
            await asyncio.sleep(0.01)  # let every batch get this far first
            if mode == "tool":
                raise tool_choice_error()
            content = kwargs["messages"][0]["content"]
            return tool_use_message([f"i{k}" for k in range(20) if f'"id": "i{k}"' in content])

    analyzer = LLMAnalyzer(live_settings(llm_batch_size=5, llm_max_attempts=3))
    analyzer.client = SimpleNamespace(messages=RacingClient())
    result = await analyzer.analyze(items(20))

    assert seen.count("tool") >= 2, "test is only meaningful if batches raced"
    assert len(result.analyses) == 20, "every batch must recover after the downgrade"
    assert not result.failed_ids


async def test_truncated_response_is_an_error_not_a_silent_empty_result():
    """max_tokens mid tool-call returns 200 with an empty argument object.

    Parsed naively that reads as 'the model analyzed nothing' and gets logged
    as a successful batch, which is how a run silently produces no themes.
    """
    truncated = SimpleNamespace(
        content=[
            SimpleNamespace(type="tool_use", name="record_analysis", input={"analyses": []})
        ],
        usage=SimpleNamespace(input_tokens=2587, output_tokens=2000),
        stop_reason="max_tokens",
    )
    analyzer, _ = analyzer_with([truncated], llm_max_attempts=1)
    with pytest.raises(LLMError) as caught:
        await analyzer._analyze_batch(items(25), "")
    assert "max_tokens" in caught.value.message
    assert caught.value.detail["batch_size"] == 25


async def test_empty_tool_payload_is_rejected():
    """A tool call with no entries is never a valid answer to 'analyze these'."""
    empty = SimpleNamespace(
        content=[
            SimpleNamespace(type="tool_use", name="record_analysis", input={"analyses": []})
        ],
        usage=SimpleNamespace(input_tokens=100, output_tokens=10),
        stop_reason="end_turn",
    )
    analyzer, _ = analyzer_with([empty], llm_max_attempts=1)
    with pytest.raises(LLMError, match="no analyses"):
        await analyzer._analyze_batch(items(3), "")


async def test_default_token_budget_covers_the_default_batch_size():
    """Guards the pairing that truncated on the first real run.

    ~90 output tokens per analysis record is the observed cost; the budget has
    to clear that for a full batch with headroom.
    """
    s = Settings(anthropic_api_key="sk-test")
    assert s.llm_max_tokens >= s.llm_batch_size * 90 * 2
