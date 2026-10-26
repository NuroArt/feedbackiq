# FeedbackIQ

LLM-assisted customer feedback triage, built as a production-shaped service rather than a notebook.

Raw feedback goes in — Zendesk tickets, NPS comments, in-app messages, a CSV export. The service classifies each item against a strict schema, clusters the items into themes, ranks them by a deterministic priority score, files a GitHub issue for anything above the severity threshold, and posts a digest to Slack. One POST replaces a weekly triage meeting.

```
POST /v1/feedback/analyze  ->  classify (LLM)  ->  cluster  ->  rank
                                                                 |
                                       GitHub issues  <----------+----------> Slack digest
```

## Classification quality is measured, not asserted

Prompt changes are otherwise unreviewable — a diff shows the words that changed, not whether the output got worse. A 50-item hand-labelled golden set puts a number on it, and CI gates on that number.

```
## Golden-set eval — PASS
claude-sonnet-5-5 · 50 items · 3 repeats · 21.3s · 35,463 tokens

| Metric               | Score | Spread | Threshold |   |
| -------------------- | ----: | -----: | --------: | - |
| sentiment_accuracy   | 0.980 | ±0.000 |      0.94 | ✅ |
| category_accuracy    | 0.980 | ±0.000 |      0.92 | ✅ |
| category_macro_f1    | 0.986 | ±0.000 |      0.92 | ✅ |
| severity_within_1    | 1.000 | ±0.000 |      0.96 | ✅ |
| severity_mae         | 0.280 | ±0.140 |      0.55 | ✅ |
| theme_ari            | 0.966 | ±0.000 |      0.88 | ✅ |
```

One misclassification in fifty. Metrics are chosen for the shape of each label: **macro-F1** rather than accuracy, because a model that stopped predicting the four-item `docs` class would lose eight points of accuracy and look merely mediocre; **MAE and within-1** for severity, because it is ordinal and predicting 4 for a 5 is nearly right; **Adjusted Rand Index** for themes, because the model invents its own vocabulary and only the partition can be scored.

Two gates. An absolute floor, and a baseline diff that catches the slow slide — quality erodes three points at a time across a dozen harmless-looking tweaks, not off a cliff. `make eval-stub` runs the same harness for free against the offline analyzer.

## Status

Runs end to end against live services.

| | |
| --- | --- |
| LLM | 50-item golden set, macro-F1 **0.986**, ARI **0.966**, baseline committed |
| GitHub | issues filed, matched and commented — idempotent across six theme re-wordings |
| Slack | digests posted |
| Tests | **111**, hermetic, environment-isolated |
| CI/CD | test → scan → smoke → deploy, plus a nightly eval with a regression gate |

Two honest caveats: the Dockerfile and Terraform were authored outside a Docker/AWS environment, so build and `terraform validate` before trusting them in CI; and 50 items at near-ceiling scores mostly demonstrates the set is not yet hard enough to discriminate between good models — it earns its keep when it grows from real production misclassifications.

`ARCHITECTURE.md` explains the design decisions, where the AI sits in the workflow, and the nine defects that running it against live services exposed.

---

## Quickstart

No API keys required. With none configured, the service runs a deterministic keyword analyzer that satisfies the same interface, so the whole thing works offline.

```bash
pip install -r requirements-dev.txt
make dev                      # http://localhost:8000
```

Open `http://localhost:8000` for the operator UI, or:

```bash
curl -X POST localhost:8000/v1/feedback/analyze \
  -H 'content-type: application/json' \
  -d @samples/request.json | jq '.executive_summary, .themes[0]'
```

Going live is three environment variables. Put them in `.env` rather than your
shell — the app reads it, and the test suite deliberately ignores it:

```bash
cp .env.example .env
# ANTHROPIC_API_KEY=...        -> real classification
# GITHUB_TOKEN= / GITHUB_REPO= -> issues get filed
# SLACK_WEBHOOK_URL=...        -> digest gets posted
```

`DRY_RUN=true` (or `"dry_run": true` per request) runs the full pipeline and reports exactly what it *would* write, without writing it.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/feedback/analyze` | Run the workflow over a JSON batch |
| `POST` | `/v1/feedback/analyze-csv` | Same, from a Zendesk/Typeform-style CSV export |
| `GET` | `/healthz` | Liveness |
| `GET` | `/readyz` | Readiness + which integrations are wired |
| `GET` | `/metrics` | Prometheus text format |
| `GET` | `/docs` | OpenAPI UI (generated from the Pydantic models) |

Workflow routes take an `X-API-Key` header when `API_KEY` is set; ops routes never do, so probes stay simple.

## Layout

```
app/
  config.py           env-driven settings, with redacted() for safe startup logging
  models.py           the contract at all three boundaries (HTTP, LLM, integrations)
  errors.py           typed errors -> one HTTP status each
  observability.py    Prometheus metrics, request-id middleware, stage timers
  logging_config.py   structlog JSON logging
  deps.py             auth dependency
  routers/api.py      routes
  services/
    llm.py            forced-tool-use analyzer + offline stub, retries, token metering
    github_client.py  idempotent issue upsert (body-marker listing + similarity)
    theme_matching.py cross-run theme matching: normalization, Jaccard, threshold
    slack_client.py   Block Kit digest via incoming webhook
    pipeline.py       orchestration, clustering, ranking, graceful degradation
  static/index.html   operator UI
evals/
  golden_set.jsonl    50 hand-labelled items with gold theme clusters
  scoring.py          accuracy, macro-F1, severity MAE, Adjusted Rand Index
  runner.py           repeats, baseline diffing, JSON + markdown report, exit codes
  thresholds.json     per-profile floors and regression tolerance
  baseline.json       committed scores the next run is diffed against
scripts/
  preflight.py        five-step live readiness check, fails fast and cheap
  e2e.py              end-to-end against real GitHub + Slack, proves dedup
  init-history.sh     builds the initial commit history
  setup-hooks.sh      installs the credential-scanning pre-commit hook
docs/LIVE_RUN.md      the first-live-run procedure
tests/                111 tests, hermetic (env-isolated: exported app config
                      and a local .env are both stripped per test)
infra/main.tf         ECS Fargate + ALB + Secrets Manager + alarms
ops/prometheus.yml    local scrape config and starter alert rules
.github/workflows/    ci.yml  test -> build+scan+smoke -> deploy
                      eval.yml nightly golden-set eval with a regression gate
```

## Commands

```bash
make preflight     # live readiness check (~1,300 tokens)
make e2e           # dry run through the full pipeline, no writes
make e2e-live      # writes to GitHub + Slack, runs twice to prove dedup
make test          # pytest, offline, no spend
make lint          # ruff + mypy
make eval-stub     # golden-set eval, stub profile, free
make eval          # golden-set eval against the real model (costs tokens)
make eval-baseline # accept current scores as the new baseline
make docker        # build the image
make smoke         # hit a running instance with the sample payload
```

After cloning, install the pre-commit hook that scans staged files for
credential patterns:

```bash
scripts/setup-hooks.sh
```

It is a backstop against the realistic accident — pasting a key into a config
file — not a security control. The real protections are `.gitignore` and never
putting secrets in tracked files.

## Evaluating classification quality

`--repeats` is not optional for a real verdict: one pass of a stochastic model is a sample, not a score. Three passes give the spread that `regression_tolerance` is calibrated against, and `severity_mae` moves far more between runs (±0.14) than the accuracy metrics (±0.00), so it carries its own tolerance.

The nightly workflow runs this against the real model, posts the table to the job summary, opens an issue tagged `eval-regression` on failure, and keeps 90 days of reports. On a PR, apply the **`run-eval`** label and the result is posted as a comment that updates in place. Baselines update only by explicit action — an auto-updating baseline ratchets downward in silence and defeats the point.

See `evals/README.md` for the labelling rules and why specific adversarial items are in the set.

## It has actually been run

Against a real repository and a real Slack workspace, not a mock.

The failure mode this had to survive: **the model names the same theme
differently on every run.** Across one session it produced six names for the
same two problems —

```
CSV upload crashes losing data   /  and loses data  /  and data loss
                                 /  causing data loss  /  with data loss
Export fails with server error   /  Report export fails with error  /  with errors
```

A fingerprint built from the theme name produced a different hash each time,
so a job designed to run every morning filed a fresh set of duplicates every
morning. Matching on normalized content words instead, all six variants resolve
to the same issue: scores ran 0.60–1.00 against each other and 0.00 against
unrelated themes.

Proof is `make e2e-live`, which runs the pipeline twice and fails the build if
the second run files anything:

```
run 1   github_candidates_loaded  count=8
        comment_on_issue   Matched existing issue #18 (similarity 1.00)
        comment_on_issue   Matched existing issue #19 (similarity 0.60)
        comment_on_issue   Matched existing issue #20 (similarity 1.00)
        comment_on_issue   Matched existing issue #21 (similarity 1.00)
        post_digest        Posted digest with 8 themes.

run 2   (same four matches, zero issues created)

PASS  idempotency holds: run 2 matched 4 existing issue(s) and filed nothing new.
```

Every match records its similarity score in the issue comment, so a reviewer
can audit why two themes were treated as one rather than take it on trust.

The generated issues are public: **[NuroArt/feedbackiq-demo](https://github.com/NuroArt/feedbackiq-demo/issues)**
— four issues for four themes, each carrying comments from successive runs.

## Known gaps

Honest list, because a "production-ready" claim with no caveats is a red flag:

- **No persistence.** Runs are stateless; results return in the response and go to GitHub/Slack. A `runs` table (Postgres + Alembic) is the first thing to add for history and trend lines.
- **Synchronous.** A 500-item batch takes minutes on one HTTP request. The next iteration returns `202` with a run id and moves execution to a worker (SQS + a consumer, or Celery).
- **No deduplication across runs** beyond the GitHub fingerprint. Feeding the same CSV twice re-analyzes it.
- **Traces are approximated** by per-stage timers rather than real spans — see ARCHITECTURE.md for the OpenTelemetry swap.
- **Docker and Terraform are unvalidated** in this environment.
- **The live scores are from one model on one day.** They are excellent, but 50 items at ceiling performance mostly proves the set isn't hard enough yet to discriminate between good models. The set earns its keep when it starts catching real production misclassifications.
- **50 items is a smoke test, not a benchmark.** It catches gross regressions. Catching subtle ones needs a few hundred items, drawn from real misclassifications rather than written in one sitting.
