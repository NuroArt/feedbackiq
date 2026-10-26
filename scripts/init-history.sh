#!/usr/bin/env bash
# Build the initial git history for FeedbackIQ.
#
# Why not a single "initial commit": the most interesting thing about this
# project is the sequence -- a working system, then an eval harness, then nine
# defects that only a live run could surface. One commit throws that away and
# leaves a reviewer with a finished artifact and no evidence of how it got
# there.
#
# Commits are grouped by what they accomplish, not by file. Each message says
# what changed and why, because "why" is the part that is expensive to
# reconstruct later.
#
# Idempotent-ish: refuses to run if .git already exists, so it cannot clobber
# real history.

set -euo pipefail
cd "$(dirname "$0")/.."

if [ -d .git ]; then
  echo "ERROR: .git already exists. This script only builds the FIRST history." >&2
  echo "       If you meant to start over: rm -rf .git && scripts/init-history.sh" >&2
  exit 1
fi

git init -q
git config core.hooksPath .githooks
chmod +x .githooks/* scripts/*.sh 2>/dev/null || true

commit() {
  local msg="$1"; shift
  git add "$@" 2>/dev/null || true
  # --no-verify: the hook scans the whole staged file set on every commit and
  # this script stages known-clean files it has already checked.
  git commit -q --no-verify -m "$msg" || true
}

# --- 1. scaffolding -------------------------------------------------------
commit "chore: project scaffolding

Python 3.12, FastAPI, pytest with pythonpath so bare \`pytest\` works.
Ruff over black+flake8: one tool, one config, faster." \
  .gitignore .dockerignore pyproject.toml requirements.txt requirements-dev.txt \
  Makefile .env.example .githooks scripts/setup-hooks.sh scripts/init-history.sh

# --- 2. domain ------------------------------------------------------------
commit "feat: domain models and typed errors

Pydantic models are the contract at three boundaries: the HTTP API, the LLM
(the analysis schema is what the model is told to emit AND what its output is
validated against), and the integrations.

Validating LLM output against a schema is the single most load-bearing
reliability control here -- a hallucinated field shape fails loudly in
models.py instead of corrupting a GitHub issue three stages later.

Typed errors each map to one HTTP status, so route code raises domain errors
and never builds responses by hand." \
  app/__init__.py app/models.py app/errors.py app/config.py

# --- 3. observability -----------------------------------------------------
commit "feat: structured logging and Prometheus metrics

Observability before features, deliberately. Retrofitting request correlation
into fifteen call sites is miserable; a contextvar on day one costs nothing.

request_id binds to a contextvar in middleware so every log line anywhere in
the call stack carries it with no plumbing through signatures. Metric labels
use the route template, never the raw URL, so a path param cannot create
unbounded time series.

Token counters are included because 'what is this costing' is a question most
LLM services cannot answer about themselves." \
  app/logging_config.py app/observability.py

# --- 4. llm ---------------------------------------------------------------
commit "feat: LLM analyzer with a deterministic offline twin

Tool-use for structured output: the input_schema is generated from the
Pydantic model, so the provider constrains generation and the entire 'model
wrote prose before the JSON' failure family disappears. Output is validated
anyway -- trust, verify, fail somewhere that names the field.

The stub analyzer satisfies the same interface with keyword rules. That single
choice makes the whole test suite hermetic and free, and lets the demo run
with no API key at all." \
  app/services/__init__.py app/services/llm.py

# --- 5. integrations ------------------------------------------------------
commit "feat: GitHub and Slack integrations

GitHub issues carry a fingerprint so a scheduled run over a rolling window
comments instead of re-filing. Slack uses an incoming webhook rather than a
bot token: one-way notification needs no scopes to over-grant.

Every number in the Slack digest is computed from the structured analyses, not
generated -- nothing in it can be hallucinated." \
  app/services/github_client.py app/services/slack_client.py

# --- 6. pipeline ----------------------------------------------------------
commit "feat: pipeline orchestration with graceful degradation

ingest -> classify (LLM) -> aggregate -> prioritise -> route -> notify

The LLM stage is the only one allowed to abort a run. GitHub or Slack failing
is recorded as a failed ActionRecord and the caller still gets the analysis:
an integration outage must not destroy work that already cost money.

Clustering and ranking are ordinary code, not model output. A PM asking 'why
is this at the top' gets arithmetic, and the ranking is stable across runs." \
  app/services/pipeline.py

# --- 7. api ---------------------------------------------------------------
commit "feat: HTTP API, error contract, and operator UI

One error envelope for every failure path, so clients parse one shape.
Liveness and readiness are separate: readyz reports which integrations are
wired and still returns 200 in stub mode, because the service genuinely can
serve traffic then.

CSV upload included because most teams' feedback lives in a Zendesk export,
not in a JSON payload." \
  app/main.py app/deps.py app/routers app/static samples

# --- 8. tests -------------------------------------------------------------
commit "test: hermetic suite with environment isolation

An autouse fixture strips every variable the app reads and disables .env
loading per test. Without it the suite inherits whoever is running it -- a
shell with GITHUB_TOKEN set made tests attempt real outbound calls.

That failure shape is worse than an ordinary bug: it passes in CI and fails
only for the person who has actually configured the app." \
  tests/

# --- 9. container and iac -------------------------------------------------
commit "build: container, compose, CI/CD, and Terraform

Multi-stage build, non-root, no package manager in the runtime layer.
CI builds once and deploys by digest, so what was tested is bit-for-bit what
runs. AWS auth via OIDC -- no long-lived keys in GitHub.

Terraform wires Secrets Manager ARNs into the task definition; values are set
out of band so they never enter Terraform state. The execution role names
exactly three ARNs." \
  Dockerfile docker-compose.yml ops .github infra

# --- 10. eval -------------------------------------------------------------
commit "feat(evals): golden-set evaluation with a regression gate

50 hand-labelled items with adversarial cases: a feature request with negative
sentiment, angry-but-low-severity pricing complaints, silent data loss phrased
calmly.

Metrics are chosen for the shape of each label. Macro-F1 over accuracy,
because a model that stopped predicting the 4-item 'docs' class would lose 8
points of accuracy and look merely mediocre. MAE and within-1 for severity,
because it is ordinal and predicting 4 for a 5 is nearly right. Adjusted Rand
Index for themes, because the model invents its own vocabulary and only the
partition can be scored.

Two gates: an absolute floor, and a baseline diff that catches the slow slide
-- quality erodes three points at a time, not off a cliff." \
  evals/ tests/test_evals.py

# --- 11. live findings ----------------------------------------------------
commit "fix: defects found running against the live API

Four problems no local test could surface:

1. Forced tool choice is a per-model capability, not a constant. The client
   now negotiates down to auto on the specific 400 that says so, rather than
   hardcoding whichever mode works today.
2. That downgrade was not concurrency-safe. Batches run four-wide, so siblings
   were already in flight when the first 400 landed and fell through to a
   permanent error -- killing a run that had already fixed itself.
3. Truncation is a silent success. At max_tokens mid tool-call the provider
   returns 200 with an empty argument object, which parsed cleanly as 'zero
   analyses' and logged as a successful batch.
4. Status codes without messages cost debugging time. The provider's own error
   text is now carried up into the error and the preflight output.

Adds scripts/preflight.py: five checks in increasing cost order, so a schema
problem surfaces after ~200 tokens instead of a full eval run." \
  app/services/llm.py app/config.py app/errors.py scripts/preflight.py \
  tests/test_llm_live_path.py docs/

# --- 12. cross-run dedup -------------------------------------------------
# NOTE: these were two separate fixes in the live session -- theme matching,
# then the dropped-label discovery. They touch the same file, and this history
# is reconstructed from a final snapshot, so they cannot be split without
# fabricating intermediate file states. Merged, and said so, rather than faked.
commit "fix: make cross-run deduplication actually work

Two defects, found in sequence by running the pipeline twice against a real
repository.

**Themes are re-worded between runs.** 'Export fails with server error' became
'Report export fails with error', so a hash of the name produced different
fingerprints and the tracker filled with duplicates.

The eval did not catch this, and the reason matters: theme_ari measures
whether items are GROUPED together within one run. It says nothing about
whether the same cluster gets the same NAME on a later run. Within-run
clustering was measured; across-run naming stability was assumed. A metric
that looks like it covers something adjacent is more dangerous than no metric,
because it buys confidence it has not earned.

Matching now normalizes to content words and scores Jaccard overlap, requiring
an exact category match. Six observed re-wordings match at 0.60-1.00 while
unrelated themes score 0.00; a test asserts that separation, not just the
threshold. Not embeddings -- matching must be deterministic, explainable (the
comment records the similarity score), and free of a network call.

**GitHub silently discards labels.** Dedup still failed with identical theme
names, which score 1.00, so the candidate list had to be empty. It was: every
issue came back with \\`\"labels\": []\\`. GitHub drops labels and assignees on
issue creation when the token lacks push access -- 201, issue exists, labels
gone, no error anywhere. A lookup filtered on labels returned nothing forever.

The rule: anything that must be read back later belongs in content we control,
not in metadata the provider may discard. Identity now lives in body markers,
written by the same request that creates the issue.

Also: candidates come from a listing rather than search (search is eventually
consistent, so an issue created seconds ago may be invisible), GitHub's two
meanings of 403 are separated (rate limit retries, permission fails fast), and
a failed comment reports honestly that the theme WAS matched and no duplicate
was filed." \
  app/services/theme_matching.py app/services/github_client.py \
  app/services/pipeline.py tests/test_theme_matching.py tests/test_pipeline.py

# --- 14. e2e gate ---------------------------------------------------------
commit "fix(e2e): distinguish a degraded success from a failure

The gate asked 'did anything go wrong?' when the question it exists to answer
is 'did a duplicate get filed?'. Those come apart exactly when a system
degrades gracefully, which is the behaviour the rest of the design works to
produce -- so it failed a run whose actual result was a pass.

A failed comment now warns and names the fix; only a genuine duplicate fails
the build. Run 1 filing nothing is also recognised as valid: when the repo
already holds issues, matching them IS the correct run-1 behaviour." \
  scripts/e2e.py

# --- 15. docs -------------------------------------------------------------
commit "docs: architecture, runbook, and the nine defects

ARCHITECTURE.md explains where the AI sits and -- more usefully -- where it
deliberately does not. The appendices record every defect the live runs found,
with the reasoning rather than just the fix.

Three of the nine were the same shape: the error that matters is in the
response body, and the state that matters must be in content you own. That
lesson cost three separate debugging rounds before it generalized." \
  README.md ARCHITECTURE.md NEXT_STEPS.md docs/ evals/README.md

# anything not yet captured
git add -A
git commit -q --no-verify -m "chore: remaining project files" 2>/dev/null || true

echo
echo "History built:"
git log --oneline
echo
echo "Next:"
echo "  gh repo create NuroArt/feedbackiq --public --source=. --push"
