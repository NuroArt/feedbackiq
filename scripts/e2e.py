#!/usr/bin/env python
"""End-to-end integration check against real GitHub and Slack.

    python scripts/e2e.py                 # dry run: shows what it WOULD do
    python scripts/e2e.py --live          # files issues, posts to Slack
    python scripts/e2e.py --live --twice  # the one that proves idempotency

Runs the pipeline in-process, so no server, no curl, no editing JSON by hand.

`--twice` is the point of this script. The service is meant to run on a
schedule over a rolling window, so the same feedback arrives in consecutive
runs. The naive implementation files the same issue every morning. Each issue
carries a fingerprint and the client searches for it before creating -- but
GitHub's search index lags issue creation by a few seconds, which means the
dedup can pass a unit test and still duplicate in production. The only honest
check is to run it twice against the real API and look at what happened.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Settings
from app.models import AnalyzeRequest
from app.services.pipeline import FeedbackPipeline

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
OK, FAIL, WARN = f"{GREEN}PASS{RESET}", f"{RED}FAIL{RESET}", f"{YELLOW}WARN{RESET}"
SAMPLE = Path(__file__).parent.parent / "samples" / "request.json"

# Dedup used to depend on GitHub's search index, which is eventually
# consistent, and the script waited 20s so a second run would not report a
# false duplicate. Candidates now come from a label listing, which is
# read-after-write consistent, so the wait exists only to keep the two runs
# visibly distinct in the log.
SEARCH_INDEX_DELAY = 3


def show_config(settings: Settings, live: bool) -> bool:
    print(f"\n{DIM}FeedbackIQ end-to-end{RESET}\n")
    print(f"  mode:    {'LIVE -- this writes to real systems' if live else 'dry run'}")
    print(f"  llm:     {'live' if settings.llm_enabled else 'offline stub'}")
    repo = settings.github_repo if settings.github_enabled else "not configured"
    print(f"  github:  {repo}")
    print(f"  slack:   {'configured' if settings.slack_enabled else 'not configured'}")
    print(f"  severity threshold: {settings.github_severity_threshold}")

    if live and not (settings.github_enabled or settings.slack_enabled):
        print(f"\n  {FAIL}  --live with no integrations configured. Nothing to prove.")
        print(f"  {DIM}      Set GITHUB_TOKEN + GITHUB_REPO and/or SLACK_WEBHOOK_URL.{RESET}")
        return False
    return True


def report(label: str, response) -> None:
    print(f"\n{DIM}--- {label} ---{RESET}")
    print(
        f"  run_id {response.run_id}  ·  {response.metrics.items_analyzed} items  "
        f"·  {response.metrics.themes} themes  "
        f"·  {response.metrics.total_duration_ms:.0f}ms"
    )
    print(f"\n  {response.executive_summary}\n")

    for theme in response.themes[:5]:
        flag = "→ issue" if theme.max_severity >= 3 else ""
        print(
            f"  {DIM}{theme.priority_score:>6}  sev{theme.max_severity}  "
            f"x{theme.item_count}  {theme.name}  {flag}{RESET}"
        )

    print()
    for a in response.actions:
        mark = {"created": GREEN, "failed": RED, "dry_run": YELLOW}.get(a.status, DIM)
        print(f"  {mark}{a.status:<8}{RESET} {a.provider:<7} {a.action:<18} {a.detail or ''}")
        if a.target:
            print(f"           {DIM}{a.target}{RESET}")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live", action="store_true", help="Actually write to GitHub and Slack."
    )
    parser.add_argument(
        "--twice",
        action="store_true",
        help="Run twice and verify the second run does not duplicate.",
    )
    args = parser.parse_args()

    settings = Settings()
    if not show_config(settings, args.live):
        return 2

    payload = json.loads(SAMPLE.read_text())
    payload["dry_run"] = not args.live
    request = AnalyzeRequest.model_validate(payload)

    pipeline = FeedbackPipeline(settings)
    first = await pipeline.run(request)
    report("run 1", first)

    # Separate "the thing under test broke" from "something adjacent broke".
    # A comment that could not be posted is a degraded outcome, not a failed
    # dedup -- the theme WAS matched and no duplicate was filed. Treating the
    # two the same aborted a run whose actual result was a pass.
    failures = [a for a in first.actions if a.status == "failed"]
    fatal = [a for a in failures if a.action != "comment_on_issue"]
    degraded = [a for a in failures if a.action == "comment_on_issue"]

    if degraded:
        print(
            f"\n  {WARN}  {len(degraded)} comment(s) could not be posted. "
            f"Dedup still worked -- no duplicates were filed."
        )
        print(f"  {DIM}      {degraded[0].detail[:160]}")
        print(
            f"        Add 'Contents: Read and write' to the token to grant push access.{RESET}"
        )

    if fatal:
        print(f"\n  {FAIL}  {len(fatal)} action(s) failed:")
        for a in fatal:
            print(f"  {DIM}      {a.provider}: {a.detail}{RESET}")
        return 1

    if not args.twice:
        if args.live:
            print(f"\n  {OK}  run 1 complete. Re-run with --twice to prove idempotency.")
        else:
            print(f"\n  {OK}  dry run complete. Add --live to write for real.")
        return 0

    if not args.live:
        print(f"\n  {WARN}  --twice is meaningless without --live (nothing is written).")
        return 0

    # Run 1 may legitimately file nothing -- if the repo already holds issues
    # from an earlier session, run 1 matches them instead. Either outcome
    # leaves issues for run 2 to deduplicate against, so count both.
    issue_urls = {
        a.target
        for a in first.actions
        if a.provider == "github"
        and a.target
        and a.action in ("create_issue", "comment_on_issue")
    }
    print(f"\n{DIM}Pausing {SEARCH_INDEX_DELAY}s, then running again.")
    print("Dedup reads an issue listing, which is read-after-write consistent,")
    print(f"so no wait is needed for correctness.{RESET}")
    await asyncio.sleep(SEARCH_INDEX_DELAY)

    second = await pipeline.run(request)
    report("run 2", second)

    # The verdict: run 2 must comment, not create.
    new_issues = [
        a
        for a in second.actions
        if a.provider == "github" and a.status == "created" and a.action == "create_issue"
    ]
    comments = [
        a for a in second.actions if a.provider == "github" and a.action == "comment_on_issue"
    ]
    matched = [a for a in comments if "Matched existing issue" in (a.detail or "")]

    print()
    if not issue_urls:
        print(f"  {WARN}  run 1 filed no issues, so there was nothing to deduplicate.")
        print(f"  {DIM}      Lower GITHUB_SEVERITY_THRESHOLD and try again.{RESET}")
        return 0
    if new_issues:
        print(f"  {FAIL}  idempotency broken: run 2 filed {len(new_issues)} NEW issue(s).")
        for a in new_issues:
            print(f"  {DIM}      {a.target}{RESET}")
        print(f"  {DIM}      Theme matching failed. Likely causes: the model re-worded")
        print("        a theme past the similarity threshold, or the issue body lost")
        print("        its feedbackiq-theme marker. Compare the theme names in the")
        print(f"        two runs above.{RESET}")
        return 1

    print(
        f"  {OK}  idempotency holds: run 2 matched {len(matched)} existing "
        f"issue(s) and filed nothing new."
    )
    if len(matched) > len([c for c in comments if c.status == "created"]):
        print(
            f"  {DIM}      (some comments could not be posted; the match is what "
            f"prevents duplicates){RESET}"
        )
    print(f"\n  {DIM}Check the repo: each issue should have one comment from run 2,")
    print(f"  and the issue count should be unchanged.{RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
