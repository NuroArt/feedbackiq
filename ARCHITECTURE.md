# Architecture

## The workflow being automated

A product team receives feedback through four or five channels. Someone — usually a PM — reads it weekly, notices the same complaint three times, and files a ticket. That job is pattern recognition over unstructured text, which is what an LLM is genuinely good at, wrapped in routing and prioritisation, which is what an LLM should not be trusted with.

FeedbackIQ splits the job along exactly that line.

```
 ingest ──▶ classify ──▶ aggregate ──▶ prioritise ──▶ route ──▶ notify
  JSON        LLM         code          code          GitHub    Slack
  or CSV   (per item)   (cluster)     (score)        (issue)   (digest)
            ▲ the only non-deterministic stage
```

## Where the AI sits, and where it deliberately does not

**The LLM does one thing:** convert one piece of unstructured text into one structured record — `{sentiment, category, severity, summary, theme}`. That is a judgement about language, which is the task LLMs are reliable at.

**Everything downstream is ordinary code.** Clustering is a dictionary keyed on the normalised `theme` string. Ranking is arithmetic:

```python
priority = item_count + 2 * max_severity + 1.5 * enterprise_count + 3 * negative_share
```

**Why this split matters:**

1. *Explainability.* A PM asking "why is this at the top?" gets an arithmetic answer, not "the model thought so."
2. *Stability.* The same input produces the same ranking on every run. A model that ranks directly will reorder its own top three between calls, which erodes trust in a daily digest faster than any single wrong answer.
3. *No hallucinated numbers.* Every figure in the Slack digest and the GitHub issue body — report counts, severity, percentages — is computed from the structured records. The model cannot invent a statistic because it is never asked for one.
4. *Cost.* One cheap call per batch of 25 items. Ranking a thousand themes costs nothing.

The honest trade: string-keyed clustering is crude. Two themes worded differently stay apart. The prompt pushes hard on reusing identical wording, which gets most of the way there; the real fix is embedding the theme strings and clustering on cosine similarity, which is the first upgrade on the list.

## Reliability controls on the LLM stage

The LLM is the only stage that can fail in ways that are novel rather than merely operational. Four controls, in order of how much they carry:

**1. Tool use for structured output.** The service declares a tool whose `input_schema` is generated from the `ItemAnalysis` Pydantic model and asks the model to call it. When the model's *input* is produced, the provider constrains it to that schema — so the entire failure family of "the model wrote a sentence before the JSON" disappears. No fence-stripping, no regex, no `json.loads` in a `try`.

There are two modes, and **which one is available is a per-model capability, not a constant**:

| Mode | Guarantee |
| --- | --- |
| `tool_choice: {"type": "tool"}` | The model **must** call the tool. |
| `tool_choice: {"type": "auto"}` | The model **may** call the tool; its input is still schema-constrained when it does. |

This was discovered the hard way on the first live run: `claude-sonnet-5-5` rejects forcing with `400 tool_choice: type "tool" and "any" are not supported for this model`. So the client **negotiates** — it starts at the configured preference (`LLM_TOOL_CHOICE`, default `forced`), and on exactly that 400 downgrades to `auto` for the remainder of the session, logs it once, and retries. The downgrade is deliberately narrow: any other 400 still fails fast without touching the mode.

Hardcoding whichever mode works today would break on the next model change, and break with an error naming a *parameter* rather than a *capability* — which is a much longer debugging session. Under `auto` the model can decline to call the tool at all; that lands in the existing `no_tool_use` retry path, which is now the documented fallback rather than a lucky accident.

**2. Validate anyway.** Constrained output still gets parsed through Pydantic. If `severity` ever arrives as `9`, the run fails in `models.py` naming the field, not three stages later inside a GitHub issue body. `tests/test_llm.py` pins this behaviour.

**3. Batch isolation.** Items are chunked and run concurrently with a semaphore of 4. A batch that fails after retries contributes its ids to `failed_ids`; the run still returns, with `items_failed` set and the shortfall stated in the executive summary. One bad batch is not a lost run.

**4. Retries that distinguish failure types.** `tenacity` with exponential backoff and jitter, but the client's own retries are disabled (`max_retries=0`) so every attempt is visible in the metrics. A 4xx that is not a 429 raises immediately — retrying a malformed request just spends money slowly.

## Graceful degradation

The expensive work happens first. GitHub or Slack failing after that must not destroy it.

| Stage | On failure |
| --- | --- |
| LLM (all batches) | Run fails, `502 llm_failure` |
| LLM (some batches) | Run succeeds, `items_failed > 0`, noted in the summary |
| GitHub | `ActionRecord(status="failed")`, pipeline continues to Slack |
| Slack | `ActionRecord(status="failed")`, response still returned |

The caller always learns what happened: `actions[]` records every attempt with `created`, `skipped`, `dry_run`, or `failed` plus a reason.

## Idempotency

The intended deployment is a scheduled run over a rolling window, so the same feedback appears in consecutive runs. Naively, that files the same issue every morning.

Each issue body carries `<!-- feedbackiq-fingerprint:<sha256 of category|theme> -->`. Before creating, the client searches the repo for that fingerprint; a hit becomes a comment on the existing issue instead. The dedup key is the theme's *identity*, not the run — so an issue closed and reopened still matches.

## Observability

Three signals, each answering a different question.

**Logs** — structlog, JSON in anything but local. `request_id` is bound to a contextvar by the middleware, so every line emitted anywhere in the call stack carries it with no plumbing through function signatures. `run_id` is bound by the pipeline on top. One grep reconstructs a run end to end:

```json
{"stage":"analyze","duration_ms":2841.3,"analyzed":10,"failed":0,
 "event":"stage_complete","request_id":"08e5…","run_id":"run_6b40a34f5b9d",
 "level":"info","timestamp":"2026-10-25T22:04:11.154Z"}
```

**Metrics** — Prometheus at `/metrics`.

| Metric | Answers |
| --- | --- |
| `http_requests_total` / `http_request_duration_seconds` | Is the service healthy? |
| `llm_calls_total{outcome}` | Is the model failing, and how? (`http_429`, `schema_invalid`, `no_tool_use`) |
| `llm_call_duration_seconds` | Is the provider slow? |
| `llm_tokens_total{direction}` | **What is this costing?** × unit price = live spend |
| `integration_calls_total{provider,outcome}` | Is GitHub or Slack broken? |
| `pipeline_stage_duration_seconds{stage}` | Which stage is the bottleneck? |
| `pipeline_in_flight` | Is work queueing up? |

Label cardinality is deliberately bounded: `path` uses the route template from `request.scope["route"]`, never the raw URL, so a path parameter can't create unbounded time series. Token metrics are the one most LLM services skip and then cannot answer a budget question about.

**Traces** — approximated by `stage_timer`, which records a histogram and emits start/finish logs. Real spans are a drop-in: `pip install opentelemetry-instrumentation-fastapi`, instrument the app at startup, and change `stage_timer`'s body to open a span. Every call site stays as it is — that's why the timing lives behind a context manager instead of inline `time.perf_counter()` calls.

## Evaluation

Everything above makes the LLM stage *operationally* reliable — it won't crash, it won't emit malformed records, it won't lose a run. None of it says whether the classifications are any **good**, and nothing in a normal test suite can: `assert sentiment == "negative"` on a stochastic model is a flaky test waiting to happen.

That's what `evals/` is for. 50 hand-labelled items, scored on four axes, gated in CI.

### Why these four metrics

**Accuracy** is the headline but hides the failure that matters. `docs` is 4 items of 50; a model that stopped predicting it entirely would lose 8 points of accuracy and look merely mediocre. **Macro-F1** weights every class equally, so that same collapse craters the score. It's the gated metric; accuracy is reported for context.

**Severity is ordinal**, so exact-match accuracy is the wrong tool — predicting 4 when gold is 5 is nearly right, predicting 1 is not, and accuracy scores both as zero. **MAE** respects the distance; **within-1** answers the operational question (would this have been routed to the right urgency bucket?). **Signed bias** is the one that reads on a dashboard: positive means the model systematically over-escalates and floods the issue tracker, negative means it under-calls real incidents. Those are different problems with different fixes, and a single accuracy number shows neither.

**Theme clustering can't be scored by string match**, because the model invents its own vocabulary — that's the design. So the scorer compares the *partition* the model's themes induce against the gold partition using the **Adjusted Rand Index**: chance-corrected, naming-independent, 1.0 for identical grouping and ≤0 for no better than random. `cluster_shape` then names the failure direction, which ARI alone doesn't make legible: a ratio above 1 means the model split one problem into near-duplicate issues; below 1 means it merged distinct problems, which is worse, because a real incident gets buried inside someone else's ticket.

### Two gates, not one

| Gate | Catches |
| --- | --- |
| **Threshold** | Absolute floor. Ship nothing worse than this. |
| **Baseline + tolerance** | The slow slide — still above the floor, measurably worse than yesterday. |

The second gate is the one that earns its keep. Quality rarely falls off a cliff; it erodes three points at a time across a dozen "harmless" prompt tweaks, each of which clears the floor. `regression_tolerance` is what keeps that gate credible — a gate that cries wolf on normal variance gets disabled within a month.

Which is also why **`--repeats` exists**. One pass of a stochastic model is a sample, not a score. Three passes give a mean and a spread, and the spread is what the tolerance is calibrated against. Reporting a single run as "the number" is the classic eval mistake: the next run moves two points and nobody can tell a regression from noise.

### The adversarial items

A golden set written to be easy measures nothing. Four items exist specifically to break naive behaviour:

- **`g022`** — a feature request with *negative* sentiment ("we evaluated a competitor because..."). Sentiment and category must be judged independently.
- **`g023`–`g026`** — angry pricing complaints that are operationally low-severity. A model reading severity off emotional intensity scores badly here, deliberately.
- **`g014`** — "the importer silently drops rows", calm wording, severity 5, because it's silent data loss.
- **`sso_login_failure`** vs the other bug clusters — four high-severity bugs that must *not* be merged into one theme.

### Failure output is a diagnosis, not a score

A failing gate prints the confusion matrix, the three weakest classes with precision/recall split, the cluster ratio, and any unreturned ids. The difference between "macro-F1 dropped to 0.68" and "`usability` is being read as `bug` six times, recall 0.14" is the difference between a number to argue about and a prompt line to fix.

### What runs where

The scorers are pure functions over label lists — no I/O, no model, no config — so all 25 harness tests run in the free suite on every push. The paid run is a separate workflow: nightly, on demand, or on a PR carrying the `run-eval` label (the signal that a prompt, the model, or the schema changed). The harness job gates the paid job: if the scorer is broken, spending tokens is pointless.

Baselines update only by explicit action. An auto-updating baseline ratchets downward in silence, which is strictly worse than having no baseline at all, because it looks like coverage.

### Honest limits

50 items catches gross regressions, not subtle ones — that needs a few hundred, grown from real production misclassifications rather than written in one sitting. The set has also never been scored against a live model in this environment, so the `live` thresholds encode what the product needs rather than what the model does; the first real run is what calibrates them and produces the `live` baseline.

## Error handling

Every failure returns the same envelope, so a client parses one shape:

```json
{"error": "llm_failure", "message": "...", "request_id": "08e5…", "detail": {...}}
```

Three handlers in `main.py` cover it: typed domain errors (each class carries its own status), validation errors (422 with the field list), and a catch-all that logs the stack trace and returns nothing internal. Route code raises domain errors and never builds an HTTP response by hand.

## Security

- **No secrets in images or task definitions.** Terraform wires Secrets Manager ARNs into the ECS `secrets` block; values are set out of band so they never enter Terraform state. The execution role's read policy names exactly three ARNs.
- **No long-lived cloud credentials in CI.** GitHub Actions assumes an AWS role via OIDC.
- **Container runs as uid 10001**, from a slim runtime stage with no build toolchain and no package manager.
- **Network**: tasks in private subnets, no public IP, ingress only from the ALB security group.
- **Startup logs config through `redacted()`** — you can prove in production what a pod booted with, without a secret appearing in CloudWatch.
- **Constant-time API key comparison** (`secrets.compare_digest`).
- **Trivy scans every image**, SARIF uploaded to GitHub code scanning. Currently reports without gating; flip `exit-code: "1"` once the base image is clean.

## What I would build next

In priority order, with reasoning:

1. **Persistence** (Postgres + Alembic). Without it there are no trends, and "is this theme growing?" is the question a PM actually asks.
2. **Async execution.** `202` + run id, work moved to an SQS consumer. A 500-item batch on a synchronous request is a timeout waiting to happen.
3. **Embedding-based clustering.** Removes the string-match brittleness described above.
4. **Human-in-the-loop.** Slack Block Kit buttons — approve / reject / reclassify — writing back as labelled data. This is also how the golden set grows from real disagreements instead of imagined ones.
5. **Prompt versioning.** Pin the system prompt to a hash recorded in the eval report, so a quality change can be attributed to a specific prompt revision rather than inferred from commit timestamps.

*(Golden-set evaluation was #4 on this list and is now built — see the Evaluation section above.)*

## Skills this project demonstrates

**AI/LLM integration** — schema-constrained output via forced tool use, retry and failure taxonomy, token-level cost instrumentation, batching with bounded concurrency, and a clear line between what the model decides and what code decides.

**LLM evaluation** — a labelled golden set with adversarial cases, metrics chosen for the shape of each label (macro-F1 for class imbalance, MAE for ordinals, ARI for unsupervised clustering), variance measured across repeats, and a two-stage gate wired into CI so prompt changes arrive as a reviewable number.

**Workflow automation** — an end-to-end business process with two external systems, idempotency designed in rather than patched on, dry-run mode, and per-action outcome reporting.

**Platform engineering** — multi-stage non-root container, hermetic CI with scanning and a real smoke test against the built image, digest-pinned deploys, OIDC instead of static keys, least-privilege IAM, and infrastructure as code with a rollback-on-failure deployment strategy.

**Production judgement** — the things that show up at 3am: bounded metric cardinality, degradation paths that preserve expensive work, a single error contract, probes that distinguish liveness from readiness, and an offline mode that makes the test suite hermetic and free.

## Appendix: what the first live run found

Four defects, none of which any amount of local testing would have surfaced,
recorded here because the failure modes are more instructive than the fixes.

**1. Forced tool choice is not universally supported.** `claude-sonnet-5-5`
returns `400 tool_choice: type "tool" and "any" are not supported for this
model`. The service now negotiates down to `auto` rather than hardcoding
either mode. See the Evaluation and reliability sections above.

**2. The downgrade was not concurrency-safe.** Batches run four-wide, so
several were already in flight with the forced flag still set when the first
400 arrived. The downgrade helper returned "nothing to do" for the siblings,
which sent them to `LLMPermanentError` and killed a run that had already fixed
itself. The check is now idempotent: the *error* decides whether to retry, not
whether this particular call performed the mutation.

**3. Truncation is a silent success.** With `max_tokens=2000` and 25 items per
batch, the response hit the ceiling mid tool-call. The provider returns **200**
with a `tool_use` block whose arguments are empty, which parses cleanly as
"zero analyses" and logged as `llm_batch_ok ... returned=0`. A successful-
looking batch that analyzed nothing is the worst failure shape available:
nothing alerts, and the digest is simply empty. The client now checks
`stop_reason == "max_tokens"` and rejects an empty payload outright, and the
default budget is sized at roughly `batch_size × 90 × 2`.

**4. Status codes without messages cost debugging time.** `LLMPermanentError`
originally carried only the HTTP status, so a 400 whose body said exactly what
was wrong surfaced as "LLM rejected the request (400)". The provider's own
message is now carried up into the error and the preflight output.

The common thread: every one of these is a *boundary* defect -- the behaviour
of a system we do not control, under conditions (concurrency, token limits,
per-model capabilities) that only exist in production. Hermetic tests were
necessary and could not have been sufficient. What they did do is make each
fix cheap to verify: all four now have regression tests that reproduce the
original failure in about a second, and three of the four were confirmed to
fail against the pre-fix code before the fix was accepted.

### Appendix: first live scores

Run against `claude-sonnet-5-5`, 50 items, 3 repeats, 2026-10-25:

| Metric | Mean | Spread |
| --- | ---: | ---: |
| `sentiment_accuracy` | 0.980 | ±0.000 |
| `category_accuracy` | 0.980 | ±0.000 |
| `category_macro_f1` | 0.986 | ±0.000 |
| `severity_within_1` | 1.000 | ±0.000 |
| `severity_mae` | 0.280 | ±0.140 |
| `theme_ari` | 0.966 | ±0.000 |

35,463 tokens, 21 seconds. One misclassification across the whole set:
`usability` read as `bug`, which is the mobile-layout boundary the set was
built to probe and a defensible reading either way.

Three things this changed:

**Thresholds became measurements.** Every `live` floor was a guess until this
run; each now sits a few points under the measured mean with the margin sized
from that metric's observed spread.

**`severity_mae` got its own tolerance.** It moved ±0.140 across three repeats
while every accuracy metric was flat at ±0.000 — ordinal scoring is simply
noisier than categorical. A single global 3-point tolerance would have failed
the nightly job on ordinary variance, and a gate that cries wolf gets disabled
within a month. Per-metric overrides exist because this run produced the data
to justify them.

**Clustering was vindicated.** ARI 0.966 with 14 predicted themes against 12
gold means theme strings are reused reliably enough for string-keyed
aggregation. Embedding-based clustering was the most intellectually appealing
item on the roadmap; this measurement is what justifies *not* building it.

The caveat worth keeping: near-ceiling scores on 50 items mostly demonstrate
that the set is not yet hard enough to discriminate between good models. Its
value starts when it grows from real production misclassifications.

### Appendix: test isolation

A fifth defect surfaced once the integrations were configured for real: the
test suite inherited the developer's exported environment. A shell with
`GITHUB_TOKEN`, `SLACK_WEBHOOK_URL` and `LLM_TOOL_CHOICE=auto` set turned
`settings.github_enabled` true inside tests that assume it is false, and the
pipeline then tried to make real outbound calls from a unit test -- one of
them got as far as attempting a POST to `hooks.slack.com`.

Seven tests failed. None of them were wrong; the environment was.

This is worse than an ordinary bug because of its shape: the suite passes in
CI (clean environment) and fails only for the person who has actually
configured the app. The failure lands on whoever is furthest along, and it
looks like their setup is broken rather than the tests. A suite that behaves
differently depending on who runs it is worse than no suite, because it is
trusted and wrong.

An autouse fixture now strips every variable the app reads and disables
`.env` loading for the duration of each test -- both leak paths, since a
developer's real `.env` sits in the repo root where pytest runs. Two tests
guard the fixture itself, and the fix was verified by disabling the fixture
and reproducing all seven original failures.

`LLM_OFFLINE_MODE` is deliberately left forced on: that is the one piece of
environment the tests want.

### Appendix: the cross-run naming problem

The sixth and most instructive defect, found by running the pipeline twice
eight minutes apart against a real repository.

Run 1 and run 2 analyzed identical input. The model named the themes
differently:

| run 1 | run 2 |
| --- | --- |
| Export fails with server error | Report export fails with error |
| CSV upload crashes losing data | CSV upload crashes and loses data |
| Slow dashboard and chart loading | *(unchanged)* |

Dedup hashed `category + theme_name`, so two of four themes produced different
fingerprints and were filed as new issues. The tracker gained three duplicates
on the second run of a job designed to run every morning.

**The eval did not catch this, and it is worth being precise about why.**
`theme_ari` of 0.966 measures whether items that belong together are *grouped*
together within a single run. It says nothing about whether the same cluster
receives the same *name* on a later run. Those are different properties;
within-run clustering was measured, across-run naming stability was assumed.
A metric that looks like it covers something adjacent is more dangerous than
no metric, because it buys confidence it has not earned.

Two changes followed.

**Theme names became evidence rather than identifiers.** Matching normalizes
to content words (stopwords dropped, light stemming so `losing`/`loses` and
`crashes`/`crashing` collapse) and scores Jaccard overlap, requiring an exact
category match. On the observed data the populations separate cleanly --
genuine re-wordings score 0.60 to 1.00, unrelated themes score 0.00 -- and a
test asserts that separation rather than just the threshold, so a future
change that closes the gap fails loudly.

Not embeddings, deliberately. Matching has to be deterministic (the same pair
must match tomorrow), explainable (an operator must be able to see *why* two
themes merged -- the comment records the similarity score), and free of a
network call. Jaccard over stemmed tokens is all three, and if it proves too
blunt that is measurable rather than mysterious.

**Candidates now come from a label listing, not the search API.** Search is
eventually consistent, so an issue created moments earlier may be invisible to
the very next run -- the original script waited 20 seconds to paper over this.
`GET /issues?labels=...` is read-after-write consistent, which removes the
failure mode and the wait together. The listing is also fetched once per run
and newly created issues are appended to it, so two near-identical themes in
the same batch cannot both file.

A seventh defect surfaced in the same run: `POST /issues/{n}/comments`
returned 403 while issue creation succeeded with the same token. The client
reported only the status code, which is consistent with the earlier lesson and
had to be fixed twice. GitHub returns 403 for both "you lack permission" and
"you tripped a secondary rate limit" -- completely different operator
responses, distinguishable only by the body. The client now carries the
provider's message into the error and retries throttling (honouring
`Retry-After`) while failing fast on permissions.

### Appendix: the silently dropped labels

The eighth defect, and the most instructive about trusting provider metadata.

After fixing cross-run theme matching, a double-run still filed four
duplicates -- this time with *identical* theme names, which the matcher scores
at 1.00. The matcher could not be at fault, so the candidate list had to be
empty. It was:

```
--- issues WITH the label filter ---     0
--- issues WITHOUT the label filter ---  14 issues
  21 []          <- labels, on every issue
  20 []
```

**GitHub silently discards labels and assignees on issue creation when the
token lacks push access to the repository.** The POST returns 201, the issue
exists, the body is correct, and the labels are simply gone. No error, no
warning, no field in the response saying so. A dedup lookup filtered on
`labels=customer-feedback` therefore returned an empty list on every run, for
ever, and every run filed duplicates.

The rule this produces: **anything that must be read back later belongs in
content we control, not in metadata the provider may discard.** The body
markers are written by the same request that creates the issue, so they cannot
be dropped independently of it. The listing now reads those and never filters
on labels. Labels are still sent -- they are useful for human triage when the
token can apply them -- but nothing depends on them, and the client logs a
warning when a created issue comes back without them, because otherwise
nothing would ever tell the operator.

The same root cause explains the 403 on `POST /issues/{n}/comments` seen in
earlier runs: a token that can create issues but lacks push access cannot
perform several adjacent writes. A failed comment is now reported as what it
is -- the theme *was* matched and no duplicate was filed, only the note could
not be appended -- rather than surfacing as a dedup failure.

Verified by replaying the real repository state (14 issues, every one with
`"labels": []`) through the fixed client: 8 candidates recovered, all four
re-worded themes matched at 0.60 to 1.00, zero new issues filed.

Three debugging rounds, one shape of lesson each time: **the error that
matters is in the response body, and the state that matters must be in content
you own.**

### Appendix: the gate that failed a passing run

A ninth, smaller finding, and the only one that was a defect in the *test*
rather than the system.

With dedup finally working, `e2e-live` still exited non-zero. Run 1 had
matched all four themes against existing issues and filed nothing -- the
result we were after -- but the comments it tried to append returned 403, and
the script treated any failed action as fatal.

The script was asking "did anything go wrong?" when the question it existed
to answer was "did a duplicate get filed?". Those come apart precisely when a
system degrades gracefully, which is the behaviour the rest of the design
works hard to produce. A gate that cannot tell a degraded success from a
failure will, in practice, be read as noise and then ignored.

It now separates the two: a failed comment prints a warning naming the fix,
the run continues, and only a genuine duplicate fails the build. Run 1 filing
nothing is also now recognised as a valid state -- when the repository already
holds issues from an earlier session, matching them *is* the correct run-1
behaviour, and the original script misread that as "nothing to deduplicate".

The underlying permission problem was the same root cause as the dropped
labels: a fine-grained token with Issues write but no push access can create
issues and nothing adjacent. `Contents: Read and write` grants push access and
restores both labels and comments. The system works without it, which is why
the distinction between fatal and degraded had to exist in the first place.

### Appendix: a tenth instance of the same lesson

`scripts/init-history.sh` reported "History built" having created zero
commits. Git had no `user.email` configured, so every `git commit` was
rejected -- and each one ended in `|| true`, which turned fourteen failures
into silence.

That guard was there for a reason: a commit group whose files were already
captured by an earlier group stages nothing, and `git commit` exits non-zero
on an empty commit. Swallowing *that* is correct. Swallowing every other
failure along with it is not.

The fix separates the two cases. Nothing staged returns quietly; a commit that
had staged content and still failed aborts with the message. The script also
checks identity up front rather than discovering it fourteen times, and
asserts the final commit count before claiming success.

This is the same shape as three earlier defects -- GitHub dropping labels on a
201, the provider returning 200 with a truncated tool call, a 403 that only
said "403". **A failure reported as a success is worse than a loud crash**,
because nothing prompts anyone to look. Four separate instances across one
project, in code written at four different times, which is a reasonable
argument that the pattern is worth checking for deliberately rather than
hoping to notice.
