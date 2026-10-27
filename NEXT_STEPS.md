# Pick-up notes — 2026-10-25

State at the end of the session: everything works against live services.

- 111 tests passing, lint clean
- Golden-set eval: macro-F1 **0.986**, theme ARI **0.966**, baseline committed
- `make e2e-live` passes — idempotency holds across runs
- Nine defects found and fixed, each with a regression test (ARCHITECTURE.md appendices)

Your working copy is `~/feedbackiq` on WSL, Python 3.12 venv at `.venv`.
Demo repo: `NuroArt/feedbackiq-demo` (issues #18–#21 are the good ones).

---

## Done (2026-10-26)

1. ~~Git history~~ — `scripts/init-history.sh` builds 14 commits grouped by
   what they accomplish, with the reasoning in each message. Credential-
   scanning pre-commit hook in `.githooks/`, installed by `make hooks`.
2. ~~README leads with the eval~~ — the table is now the first section after
   the overview, with a screenshots section wired to `docs/images/`.
3. ~~`.env` over shell exports~~ — documented in the Quickstart.

All three are done. The screenshots were dropped in favour of linking the live
demo repository, which is better evidence than an image -- it is clickable and
the comment history is visible in full.
(filenames in `docs/images/README.md`), then commit and push them.

```bash
git add docs/images/*.png
git commit -m "docs: screenshots from the live run"
git push
```

**Token expiry:** the GitHub PAT expires **Jan 1 2027**, the Anthropic key
doesn't. Put a reminder somewhere; a 401 mid-demo is avoidable.

---

## If you want to keep building

In priority order, with the reasoning rather than just the list:

**Persistence (Postgres + Alembic).** Unblocks "is this theme growing or
shrinking?", which is the question a PM actually asks and the service cannot
currently answer at all. Also a prerequisite for async. Least differentiating
work on the list — every backend candidate has CRUD — but it unblocks the rest.

**Async execution.** `202` + run id, work on a worker, `GET /v1/runs/{id}`.
A 500-item batch on a synchronous request will hit the ALB's 60-second idle
timeout. Half-built already: `PIPELINE_IN_FLIGHT`, the stage timers and
`run_id` all exist because the design anticipated this.

**Grow the golden set from production misclassifications.** 50 items at
ceiling performance mostly proves the set isn't hard enough to discriminate
between good models yet. It earns its keep when it grows from real
disagreements — which is also the argument for human-in-the-loop Slack buttons
(approve / reject / reclassify), since that's the mechanism that feeds it.

**Add a cross-run naming-stability metric to the eval.** This one is pointed:
`theme_ari` measured within-run clustering and said nothing about whether the
same cluster gets the same name tomorrow. That gap caused the duplicate-issue
bug. Running the analyzer twice over the same input and scoring theme-name
similarity between runs would have caught it before it reached GitHub.

**Prompt versioning.** Pin the system prompt's hash into the eval report so a
quality change attributes to a specific revision instead of a commit date.
Half a day, and it makes the eval work look deliberate rather than bolted on.

What to *skip*: embedding-based clustering. ARI 0.966 and six theme re-wordings
all matching says string matching is good enough. Revisit only if a future eval
run shows `theme_ari` dropping below ~0.7.

---

## The thing to say about this project

Not that it works. That it was **run**, and that running it found nine defects
no amount of local testing would have surfaced — per-model API capabilities,
concurrency against a real service, token limits on real payloads, silently
discarded provider metadata.

Three of the nine were the same shape: *the error that matters is in the
response body, and the state that matters must be in content you own.* That
lesson cost three separate debugging rounds before it generalized. Noticing
that pattern in your own debugging is the part that's hard to fake, and it's
worth saying out loud.

The eval has a similar story. It passed at 0.986 and still missed the bug that
mattered, because it measured within-run clustering while the product depended
on across-run naming. "My metric looked adjacent to the thing I cared about,
and wasn't" is a more credible thing to have learned than a green dashboard.
