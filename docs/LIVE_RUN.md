# Live run checklist

Everything in this repo has been verified against a stub. This is the procedure
for the first run against a real model. Budget: about an hour and under $5.

Do the steps in order. Each one is cheap and catches a different failure, and
stopping at the first failure is the point — a schema problem found in step 2
costs 200 tokens, the same problem found in step 5 costs a full eval.

---

## 0. Get a key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Use a key scoped to a throwaway workspace with a spend limit on it. Nothing
here should cost more than a few dollars; a limit turns a runaway loop into an
error instead of an invoice.

## 1. Preflight (~1,300 tokens, under a minute)

```bash
python scripts/preflight.py
```

Five checks, failing fast:

| Check | Catches |
| --- | --- |
| config | key missing, wrong model name |
| schema | `$ref` leakage from the Pydantic-generated schema |
| live call | **the provider rejecting the tool definition** |
| theme reuse | the model inventing a unique theme per item |

**The theme-reuse check is the one that matters.** Six probe items form two
obvious clusters. A healthy result is 2 distinct themes. If it returns 5 or 6,
nothing crashes and the product is quietly useless — every item becomes its own
"theme", producing near-duplicate GitHub issues and a meaningless digest.

If it fails there, tighten the theme rule in `SYSTEM_PROMPT` (`app/services/llm.py`)
and re-run. That loop costs about a tenth of a cent per iteration.

**If the live call fails with a 400**, the schema was rejected. Print it:

```bash
python -c "from app.services.llm import _tool_schema; import json; print(json.dumps(_tool_schema(), indent=2))"
```

The likely culprit is `_resolve()` not inlining an enum `$ref`.

## 2. Golden-set eval (~15k tokens, a few minutes)

```bash
make eval        # 3 repeats
```

Read the output in this order:

1. **Did it pass?** The `live` thresholds are aspirational — set from what the
   product needs, never measured. Expect a fail on the first run. That is not a
   problem, it is the number you came for.
2. **The spread column.** If a metric moves more than ~0.05 across three
   repeats, raise `regression_tolerance` in `evals/thresholds.json` to match, or
   the gate will cry wolf and get disabled inside a month.
3. **`theme_ari`.** Above 0.7 means string-keyed clustering is good enough and
   embedding-based clustering is a solution looking for a problem. Below 0.5
   means it isn't, and that moves up the roadmap.
4. **`severity_bias`.** Positive means the model over-escalates and will flood
   the issue tracker; raise `GITHUB_SEVERITY_THRESHOLD` or tighten the severity
   rubric in the prompt. Negative means it under-calls real incidents, which is
   the worse direction.
5. **The confusion matrix.** Two classes being mixed is a prompt fix. Everything
   being mixed is a schema or context problem.

## 3. Calibrate and baseline

Set each `live` threshold in `evals/thresholds.json` to **measured minus a
margin**, where the margin is about twice the observed spread. Then:

```bash
make eval-baseline
git add evals/thresholds.json evals/baseline.json
git commit -m "chore(evals): calibrate live thresholds from first real run"
```

Commit the report alongside it. A baseline without the run that produced it is
a number with no provenance.

## 4. End-to-end with real integrations

Use a **throwaway GitHub repo** — this files real issues.

Create a fine-grained PAT (Settings → Developer settings → Personal access
tokens → Fine-grained), scoped to one repository, with:

- **Issues: Read and write** — create issues and read them back
- **Contents: Read and write** — grants push access

Contents looks unrelated and is easy to skip. GitHub checks push access before
applying labels and before allowing issue comments, and it fails *silently* for
labels: the issue is created, returns 201, and the labels are simply absent.
Without it you get issues with no labels and 403s on every comment. Dedup still
works either way (it reads markers in the issue body, not labels), but the
output is degraded and the cause is invisible.

Then a Slack incoming webhook from an app in a test channel.

```bash
export GITHUB_TOKEN=github_pat_...
export GITHUB_REPO=yourname/feedbackiq-demo
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

make e2e          # dry run: shows exactly what it would write
make e2e-live     # writes for real, then runs again to check dedup
```

`make e2e-live` is the one that matters. It runs the pipeline, waits 20
seconds for GitHub's search index to catch up, runs it again, and then checks
that the second run **commented on existing issues rather than filing new
ones**. It exits non-zero if duplicates appear.

That wait is not a test convenience. The dedup lookup is a search query, and
GitHub search is eventually consistent — a scheduled run firing minutes later
is fine, but a second run fired instantly can miss an issue created moments
ago and duplicate it. The script makes that constraint visible instead of
hiding it behind a passing check.

Verify by eye afterwards:

- [ ] Issues carry severity and category labels
- [ ] Issue bodies contain the signal table and the fingerprint comment
- [ ] Each issue has exactly one comment from run 2, and the issue **count**
      is unchanged
- [ ] The Slack digest rendered (Block Kit fails silently on malformed blocks)
- [ ] `curl localhost:8000/metrics | grep llm_tokens` shows real counts

If run 1 files no issues at all, `GITHUB_SEVERITY_THRESHOLD` is above every
theme in the sample — lower it to 3 and retry.

## 5. Capture the evidence

For a portfolio, this is the step that converts "should work" into "does work":

- The preflight output
- The eval table from `eval-report.md`
- A screenshot of a filed GitHub issue and the Slack digest
- The `/metrics` excerpt showing token counts

Put the eval table in the README. It is the most unusual artifact in the
project and the one least likely to appear in a comparable repo.

---

## If you only have ten minutes

Steps 0 and 1. Preflight alone removes the largest unknown in the project for
about a tenth of a cent, and everything after it is incremental.
