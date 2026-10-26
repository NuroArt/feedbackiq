"""LLM analysis service.

Design notes worth defending in a review:

1. **Structured output via forced tool use.** We don't ask for "JSON in a code
   fence" and regex it out. We declare a tool whose `input_schema` is generated
   from the Pydantic model and force the model to call it. The provider then
   constrains generation to that schema, which removes the entire class of
   "the model wrote prose before the JSON" failures.
2. **Validate anyway.** Schema-constrained output still gets parsed through
   Pydantic. Trust, verify, and fail in a place that names the field.
3. **Batching.** Feedback is chunked (`llm_batch_size`) and batches run
   concurrently with bounded parallelism. One bad batch degrades that batch
   only; the run still returns partial results with `items_failed > 0`.
4. **Offline stub.** With no API key the same interface returns deterministic
   heuristic analysis, so CI, local dev, and demos all run with zero spend and
   zero network.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import structlog
from pydantic import ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.config import Settings
from app.errors import LLMError, LLMPermanentError
from app.models import Category, FeedbackItem, ItemAnalysis, Sentiment
from app.observability import LLM_CALLS, LLM_LATENCY, LLM_TOKENS

log = structlog.get_logger(__name__)

SYSTEM_PROMPT = """\
You are a product feedback analyst for a software company. You classify raw \
customer feedback into a strict schema so downstream automation can route it.

Rules:
- Judge severity from customer impact, not from how emotional the wording is. \
5 = cannot use the product or data loss. 4 = core workflow broken. \
3 = significant friction with a workaround. 2 = minor annoyance. 1 = cosmetic.
- `theme` must be a short, reusable noun phrase (3-6 words) describing the \
underlying issue, not a restatement of this one message. Reuse the identical \
wording when two items share a root cause, so they cluster.
- `summary` is one neutral sentence. No marketing language, no speculation \
about causes you cannot see in the text.
- Analyze every item you are given. Return exactly one entry per input id.
"""


class AnalysisResult:
    """Container returned by the service (analyses + usage accounting)."""

    def __init__(
        self,
        analyses: list[ItemAnalysis],
        *,
        calls: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        failed_ids: list[str] | None = None,
    ) -> None:
        self.analyses = analyses
        self.calls = calls
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.failed_ids = failed_ids or []


def _tool_schema() -> dict[str, Any]:
    """Input schema for the forced tool call, derived from the Pydantic model."""
    item = ItemAnalysis.model_json_schema()
    # Inline the enum $defs so the provider sees a self-contained schema.
    defs = item.pop("$defs", {})

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                key = node["$ref"].rsplit("/", 1)[-1]
                return _resolve(defs[key])
            return {k: _resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_resolve(v) for v in node]
        return node

    item = _resolve(item)
    return {
        "type": "object",
        "properties": {
            "analyses": {
                "type": "array",
                "description": "One entry per input feedback item, same ids.",
                "items": item,
            }
        },
        "required": ["analyses"],
    }


def _render_batch(items: list[FeedbackItem], product_context: str) -> str:
    lines = []
    if product_context:
        lines.append(f"Product context: {product_context}\n")
    lines.append("Feedback items:")
    for it in items:
        lines.append(
            json.dumps(
                {
                    "id": it.id,
                    "source": it.source,
                    "customer_tier": it.customer_tier,
                    "text": it.text,
                },
                ensure_ascii=False,
            )
        )
    lines.append(
        "\nCall record_analysis once with one entry for each id above."
    )
    return "\n".join(lines)


def _api_message(exc: object) -> str:
    """Pull the provider's own error text out of an APIStatusError.

    `str(exc)` gives a status line; the useful sentence ("tool_choice: type
    'tool' ... not supported for this model") lives in the response body.
    """
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])[:400]
    return str(getattr(exc, "message", None) or exc)[:400]


class LLMAnalyzer:
    """Live analyzer backed by the Anthropic Messages API."""

    def __init__(self, settings: Settings) -> None:
        from anthropic import AsyncAnthropic  # imported lazily: stub needs no SDK

        self.settings = settings
        self.client = AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,  # we own retry policy, so it's observable
        )
        self.tool = {
            "name": "record_analysis",
            "description": "Record the structured analysis of each feedback item.",
            "input_schema": _tool_schema(),
        }
        # Tool-choice support is a per-model capability, not a constant. Some
        # models accept {"type": "tool"} (forced: the model MUST call it);
        # others only accept {"type": "auto"} (the model MAY call it). We start
        # at the configured preference and negotiate downward on the specific
        # 400 that says so -- see _downgrade_tool_choice. The alternative,
        # hardcoding whichever one today's model supports, breaks on the next
        # model change and fails with an error that names a parameter rather
        # than a capability.
        self._forced = settings.llm_tool_choice == "forced"

    @property
    def tool_choice(self) -> dict[str, str]:
        return (
            {"type": "tool", "name": "record_analysis"}
            if self._forced
            else {"type": "auto"}
        )

    def _downgrade_tool_choice(self, message: str) -> bool:
        """Fall back to tool_choice=auto when the model rejects forcing.

        Returns True when the error is the tool-choice capability error and the
        call is therefore worth retrying -- including when a *concurrent* batch
        already performed the downgrade.

        That distinction is the whole bug this function was rewritten for.
        Batches run concurrently, so several are in flight with `_forced` still
        true when the first 400 arrives. If this returned False once the flag
        had been cleared, every sibling batch would fall through to
        LLMPermanentError and the run would die after the downgrade had already
        fixed the problem. Idempotent, logs once, retries always.
        """
        if "tool_choice" not in message:
            return False
        if not self._forced:
            return True  # already negotiated by a sibling batch; just retry
        self._forced = False
        log.warning(
            "tool_choice_downgraded",
            model=self.settings.llm_model,
            reason=message[:200],
            effect=(
                "Falling back to tool_choice=auto. The schema still constrains "
                "the tool input, but the model may decline to call the tool; "
                "that case is handled as a retryable no_tool_use error."
            ),
        )
        return True

    async def analyze(
        self, items: list[FeedbackItem], product_context: str = ""
    ) -> AnalysisResult:
        size = max(1, self.settings.llm_batch_size)
        batches = [items[i : i + size] for i in range(0, len(items), size)]
        # Bounded concurrency: enough to hide latency, not enough to trip
        # provider rate limits on a large upload.
        sem = asyncio.Semaphore(4)

        async def run(batch: list[FeedbackItem]):
            async with sem:
                return await self._analyze_batch(batch, product_context)

        results = await asyncio.gather(
            *(run(b) for b in batches), return_exceptions=True
        )

        analyses: list[ItemAnalysis] = []
        failed: list[str] = []
        first_error: BaseException | None = None
        calls = tin = tout = 0
        for batch, res in zip(batches, results, strict=True):
            if isinstance(res, BaseException):
                failed.extend(i.id for i in batch)
                first_error = first_error or res
                log.error("batch_failed", size=len(batch), error=str(res))
                continue
            batch_analyses, usage = res
            analyses.extend(batch_analyses)
            calls += 1
            tin += usage[0]
            tout += usage[1]
            returned = {a.id for a in batch_analyses}
            failed.extend(i.id for i in batch if i.id not in returned)

        if not analyses and failed:
            # Carry the underlying cause up. "All batches failed" alone sends an
            # operator to the logs to find out it was simply a bad API key.
            cause = f"{type(first_error).__name__}: {first_error}" if first_error else None
            raise LLMError(
                f"All analysis batches failed ({cause})." if cause
                else "All analysis batches failed.",
                detail={"failed_ids": failed[:20], "cause": cause},
            )
        return AnalysisResult(
            analyses,
            calls=calls,
            input_tokens=tin,
            output_tokens=tout,
            failed_ids=failed,
        )

    async def _analyze_batch(
        self, batch: list[FeedbackItem], product_context: str
    ) -> tuple[list[ItemAnalysis], tuple[int, int]]:
        from anthropic import APIError, APIStatusError

        model = self.settings.llm_model
        attempt_no = 0

        def _retryable(exc: BaseException) -> bool:
            # LLMPermanentError is checked first because it subclasses LLMError:
            # a 400 returns the same 400 on every attempt, so retrying it only
            # spends budget to reach the same answer and hides the real cause
            # behind a retry-exhausted message.
            if isinstance(exc, LLMPermanentError):
                return False
            return isinstance(exc, APIError | LLMError | asyncio.TimeoutError)

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.settings.llm_max_attempts),
            wait=wait_exponential_jitter(initial=1, max=20),
            retry=retry_if_exception(_retryable),
            reraise=True,
        ):
            with attempt:
                attempt_no += 1
                with LLM_LATENCY.labels(model).time():
                    try:
                        msg = await self.client.messages.create(
                            model=model,
                            max_tokens=self.settings.llm_max_tokens,
                            system=SYSTEM_PROMPT,
                            tools=[self.tool],
                            tool_choice=self.tool_choice,
                            messages=[
                                {
                                    "role": "user",
                                    "content": _render_batch(batch, product_context),
                                }
                            ],
                        )
                    except APIStatusError as exc:
                        LLM_CALLS.labels(model, f"http_{exc.status_code}").inc()
                        api_message = _api_message(exc)

                        # An unsupported tool_choice is a capability mismatch,
                        # not a bad request: retry once in the supported mode.
                        if exc.status_code == 400 and self._downgrade_tool_choice(
                            api_message
                        ):
                            LLM_CALLS.labels(model, "tool_choice_downgraded").inc()
                            raise LLMError(
                                "Retrying with tool_choice=auto."
                            ) from exc

                        # Any other 4xx except 429 returns the same answer on
                        # every attempt. Carry the API's own message up: the
                        # status code alone sends an operator to the logs.
                        if 400 <= exc.status_code < 500 and exc.status_code != 429:
                            raise LLMPermanentError(
                                f"LLM rejected the request "
                                f"({exc.status_code}): {api_message}",
                                detail={
                                    "status": exc.status_code,
                                    "api_message": api_message,
                                },
                            ) from exc
                        raise

                usage = (
                    getattr(msg.usage, "input_tokens", 0) or 0,
                    getattr(msg.usage, "output_tokens", 0) or 0,
                )
                LLM_TOKENS.labels(model, "input").inc(usage[0])
                LLM_TOKENS.labels(model, "output").inc(usage[1])

                # Truncation is the silent failure mode of structured output.
                # When the response hits max_tokens mid tool-call, the provider
                # still returns 200 and a tool_use block -- with a partial or
                # empty argument object. Parsed naively that looks like "the
                # model analyzed nothing", which is indistinguishable from a
                # legitimately empty result and gets logged as success.
                if getattr(msg, "stop_reason", None) == "max_tokens":
                    LLM_CALLS.labels(model, "truncated").inc()
                    raise LLMError(
                        f"Response hit max_tokens ({self.settings.llm_max_tokens}) "
                        f"with {len(batch)} items in the batch, so the tool call "
                        f"was cut off. Raise LLM_MAX_TOKENS or lower "
                        f"LLM_BATCH_SIZE.",
                        detail={
                            "max_tokens": self.settings.llm_max_tokens,
                            "batch_size": len(batch),
                        },
                    )

                payload = next(
                    (b.input for b in msg.content if getattr(b, "type", "") == "tool_use"),
                    None,
                )
                if payload is None:
                    LLM_CALLS.labels(model, "no_tool_use").inc()
                    raise LLMError("Model did not return a tool call.")
                if not payload.get("analyses"):
                    # A tool call with no entries is never a valid answer here:
                    # the prompt asks for one entry per input id.
                    LLM_CALLS.labels(model, "empty_payload").inc()
                    raise LLMError(
                        f"Model returned a tool call with no analyses for "
                        f"{len(batch)} items."
                    )

                try:
                    analyses = [
                        ItemAnalysis.model_validate(a)
                        for a in payload.get("analyses", [])
                    ]
                except ValidationError as exc:
                    LLM_CALLS.labels(model, "schema_invalid").inc()
                    raise LLMError(
                        "LLM output failed schema validation.",
                        detail=exc.errors()[:5],
                    ) from exc

                LLM_CALLS.labels(model, "success").inc()
                log.info(
                    "llm_batch_ok",
                    model=model,
                    batch_size=len(batch),
                    returned=len(analyses),
                    attempt=attempt_no,
                    input_tokens=usage[0],
                    output_tokens=usage[1],
                )
                return analyses, usage

        raise LLMError("Unreachable: retry loop exited without result.")


# --------------------------------------------------------------------------
# Offline stub
# --------------------------------------------------------------------------
_NEG = re.compile(
    r"\b(crash|broken|bug|error|fail|slow|lag|timeout|confus|hate|worst|unusable|"
    r"lost|stuck|expensive|overpriced|frustrat|can'?t|cannot|won'?t load)\b",
    re.I,
)
_POS = re.compile(r"\b(love|great|awesome|excellent|perfect|fast|easy|thank)\b", re.I)
_RULES: list[tuple[re.Pattern[str], Category, str]] = [
    (re.compile(r"\b(crash|error|bug|broken|500|exception)\b", re.I), Category.BUG,
     "Crashes and errors in core flows"),
    (re.compile(r"\b(slow|lag|timeout|perform|freeze|hang)\b", re.I),
     Category.PERFORMANCE, "Slow load and timeout issues"),
    (re.compile(r"\b(price|pricing|cost|expensive|billing|invoice)\b", re.I),
     Category.PRICING, "Pricing and billing concerns"),
    (re.compile(r"\b(doc|documentation|guide|tutorial|readme)\b", re.I),
     Category.DOCS, "Documentation gaps"),
    (re.compile(r"\b(support|ticket|response|agent|reply)\b", re.I),
     Category.SUPPORT, "Support responsiveness"),
    (re.compile(r"\b(wish|please add|feature|would be nice|support for|integration)\b",
                re.I), Category.FEATURE_REQUEST, "Requested integrations and features"),
    (re.compile(r"\b(confus|unclear|hard to find|navigat|ux|ui|layout)\b", re.I),
     Category.USABILITY, "Confusing navigation and UI"),
]


class StubAnalyzer:
    """Deterministic keyword analyzer. Same interface, no network, no spend."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def analyze(
        self, items: list[FeedbackItem], product_context: str = ""
    ) -> AnalysisResult:
        out: list[ItemAnalysis] = []
        for it in items:
            text = it.text
            neg, pos = len(_NEG.findall(text)), len(_POS.findall(text))
            sentiment = (
                Sentiment.NEGATIVE
                if neg > pos
                else Sentiment.POSITIVE
                if pos > neg
                else Sentiment.NEUTRAL
            )
            category, theme = Category.OTHER, "General feedback"
            for pattern, cat, th in _RULES:
                if pattern.search(text):
                    category, theme = cat, th
                    break
            severity = min(5, max(1, 1 + neg + (1 if it.customer_tier == "enterprise" else 0)))
            if sentiment is Sentiment.POSITIVE:
                severity = 1
            out.append(
                ItemAnalysis(
                    id=it.id,
                    sentiment=sentiment,
                    category=category,
                    severity=severity,
                    summary=(text[:140] + "...") if len(text) > 140 else text,
                    theme=theme,
                )
            )
        LLM_CALLS.labels("stub", "success").inc()
        log.info("stub_analysis", items=len(items))
        return AnalysisResult(out, calls=0)


def build_analyzer(settings: Settings) -> LLMAnalyzer | StubAnalyzer:
    if settings.llm_enabled:
        return LLMAnalyzer(settings)
    log.warning(
        "llm_offline_mode",
        reason="no anthropic_api_key" if not settings.anthropic_api_key else "forced",
    )
    return StubAnalyzer(settings)
