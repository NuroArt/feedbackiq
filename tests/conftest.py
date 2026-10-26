import os

import pytest

# Force stub mode before app import so no test can reach a real provider.
os.environ.setdefault("LLM_OFFLINE_MODE", "true")
os.environ.setdefault("LOG_FORMAT", "console")
os.environ.setdefault("ENVIRONMENT", "local")

from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app
from app.models import FeedbackItem

# Every setting the app reads from the environment. Anything here that leaks
# into a test makes the suite depend on whoever is running it.
_APP_ENV_VARS = (
    "ANTHROPIC_API_KEY", "LLM_MODEL", "LLM_MAX_TOKENS", "LLM_TIMEOUT_SECONDS",
    "LLM_MAX_ATTEMPTS", "LLM_BATCH_SIZE", "LLM_TOOL_CHOICE",
    "GITHUB_TOKEN", "GITHUB_REPO", "GITHUB_API_BASE", "GITHUB_ISSUE_LABEL",
    "GITHUB_SEVERITY_THRESHOLD", "SLACK_WEBHOOK_URL", "API_KEY", "DRY_RUN",
)


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """Strip app config out of the environment for every test.

    Without this the suite silently inherits whatever the developer exported.
    That is not a cosmetic problem: a shell with GITHUB_TOKEN and
    SLACK_WEBHOOK_URL set turns `settings.github_enabled` true inside tests
    that assume it is false, and the pipeline then attempts real outbound
    calls. A test suite that behaves differently depending on who runs it is
    worse than no test suite, because it passes in CI and lies locally.

    LLM_OFFLINE_MODE is deliberately left set: forcing the stub is the one
    piece of environment the tests *want*.
    """
    for var in _APP_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    # A local .env is the other leak path: Settings reads it, and the tests run
    # from the repo root where a developer's real .env lives.
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    monkeypatch.setenv("LLM_OFFLINE_MODE", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    get_settings.cache_clear()
    return Settings(llm_offline_mode=True, environment="local")


@pytest.fixture
def client():
    with TestClient(create_app()) as c:
        yield c


@pytest.fixture
def items() -> list[FeedbackItem]:
    return [
        FeedbackItem(id="1", text="The export crashes with a 500 error every time.",
                     customer_tier="enterprise"),
        FeedbackItem(id="2", text="Export is broken, I get an error and lose my work."),
        FeedbackItem(id="3", text="Dashboard is painfully slow, times out most mornings."),
        FeedbackItem(id="4", text="Love the new reporting view, thank you!"),
        FeedbackItem(id="5", text="Please add a Salesforce integration, would be nice."),
    ]
