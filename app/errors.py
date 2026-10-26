"""Typed application errors.

Each one maps to a single HTTP status in main.py's handlers, so route code
raises domain errors and never constructs HTTP responses by hand.
"""

from __future__ import annotations


class FeedbackIQError(Exception):
    """Base class. `code` is the stable machine-readable string clients match on."""

    code = "internal_error"
    status_code = 500

    def __init__(self, message: str, detail: object | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class AuthError(FeedbackIQError):
    code = "unauthorized"
    status_code = 401


class LLMError(FeedbackIQError):
    """LLM call failed after retries, or returned unparseable output.

    Retryable by default: a transient provider fault or a one-off bad
    generation is worth another attempt.
    """

    code = "llm_failure"
    status_code = 502


class LLMPermanentError(LLMError):
    """An LLM failure that retrying cannot fix.

    A 400 for a malformed request, an auth failure, a model name that does not
    exist -- these return the identical error on every attempt. Retrying them
    costs latency and budget to arrive at the same answer, and (worse) buries
    the real cause under a retry-exhausted message. Kept as a subclass so the
    HTTP layer still maps it to 502 without a second handler.
    """

    code = "llm_rejected"


class IntegrationError(FeedbackIQError):
    """An outbound provider (GitHub/Slack) failed. Non-fatal by default."""

    code = "integration_failure"
    status_code = 502

    def __init__(self, provider: str, message: str, detail: object | None = None):
        super().__init__(message, detail)
        self.provider = provider
