"""Application configuration.

All runtime config comes from environment variables (12-factor). Nothing
secret is ever hard-coded or logged; `redacted()` is what gets emitted at
startup so we can prove in production what config a pod actually booted with.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- service ---
    app_name: str = "feedbackiq"
    environment: Literal["local", "dev", "staging", "prod"] = "local"
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"
    api_key: str | None = Field(
        default=None,
        description="Shared secret required in the X-API-Key header. "
        "When unset, auth is disabled (local dev only).",
    )

    # --- LLM ---
    anthropic_api_key: str | None = None
    llm_model: str = "claude-sonnet-5-5"
    # Output budget. Each analysis record costs roughly 70-90 output tokens, so
    # the floor is batch_size * 90 plus headroom. The original 2000/25 pairing
    # was under that and truncated mid tool-call on the first real run, which
    # the provider reports as a 200 with an empty argument object -- so the
    # headroom here is doing real work, not padding.
    llm_max_tokens: int = 8000
    llm_timeout_seconds: float = 60.0
    llm_max_attempts: int = 3
    llm_batch_size: int = 15
    llm_tool_choice: Literal["forced", "auto"] = Field(
        default="forced",
        description=(
            "Preferred tool-choice mode. 'forced' makes the model call the "
            "analysis tool; not every model supports it, and the client "
            "negotiates down to 'auto' automatically on the 400 that says so. "
            "Set 'auto' to skip that one wasted call on a model known not to "
            "support forcing."
        ),
    )
    llm_offline_mode: bool = Field(
        default=False,
        description="Force the deterministic stub analyzer. Auto-enabled when "
        "no anthropic_api_key is present so the demo and CI run without keys.",
    )

    # --- GitHub integration ---
    github_token: str | None = None
    github_repo: str | None = Field(
        default=None, description="owner/repo that receives generated issues"
    )
    github_api_base: str = "https://api.github.com"
    github_issue_label: str = "customer-feedback"
    # Only themes at or above this severity become issues.
    github_severity_threshold: int = 3

    # --- Slack integration ---
    slack_webhook_url: str | None = None

    # --- behaviour flags ---
    dry_run: bool = Field(
        default=False,
        description="Run the full pipeline but skip all outbound writes.",
    )

    @property
    def llm_enabled(self) -> bool:
        return bool(self.anthropic_api_key) and not self.llm_offline_mode

    @property
    def github_enabled(self) -> bool:
        return bool(self.github_token and self.github_repo)

    @property
    def slack_enabled(self) -> bool:
        return bool(self.slack_webhook_url)

    def redacted(self) -> dict[str, object]:
        """Config safe to log at startup."""
        return {
            "app_name": self.app_name,
            "environment": self.environment,
            "log_level": self.log_level,
            "llm_model": self.llm_model,
            "llm_enabled": self.llm_enabled,
            "github_enabled": self.github_enabled,
            "github_repo": self.github_repo,
            "slack_enabled": self.slack_enabled,
            "auth_enabled": bool(self.api_key),
            "dry_run": self.dry_run,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
