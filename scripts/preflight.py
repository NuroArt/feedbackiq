#!/usr/bin/env python
"""Live preflight: the cheapest possible proof that the real path works.

    python scripts/preflight.py

Runs five checks in increasing order of cost, stopping at the first failure, so
a schema problem surfaces after ~200 tokens instead of after a 50-item eval:

    1. config        - key present, model set (free)
    2. schema        - the generated input_schema is structurally valid (free)
    3. api           - a 1-item call: does the provider accept the tool? (~400 tokens)
    4. clustering    - a 6-item call: do related items get the SAME theme string?
                       (~900 tokens)  <-- the assumption the whole product rests on
    5. next steps    - what to run once this passes

Check 4 is the one worth the money. Every downstream stage -- aggregation,
ranking, issue filing -- assumes the model reuses theme wording for related
items. If it emits six unique themes, nothing crashes and the product is
quietly useless: six near-duplicate GitHub issues and a meaningless digest.
That failure is invisible to a unit test and obvious here.

Exit codes: 0 ready, 1 a check failed, 2 not configured.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Settings
from app.models import FeedbackItem
from app.services.llm import LLMAnalyzer, _tool_schema

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
OK, FAIL, WARN = f"{GREEN}PASS{RESET}", f"{RED}FAIL{RESET}", f"{YELLOW}WARN{RESET}"

# Two clusters of three. A working model gives these two distinct theme strings,
# reused verbatim within each cluster.
PROBE = [
    ("p1", "The export button throws a 500 error every time I pick a date range."),
    ("p2", "Export is broken, I get an error screen and no file downloads."),
    ("p3", "Report download fails with a server error. Three of us have tried."),
    ("p4", "Dashboard takes 25 seconds to load every morning, it used to be instant."),
    ("p5", "Everything is slow since the update, charts time out constantly."),
    ("p6", "Load times on the overview page have tripled this month."),
]


def header(n: int, title: str) -> None:
    print(f"\n{DIM}[{n}/5]{RESET} {title}")


def check_config(settings: Settings) -> bool:
    header(1, "Configuration")
    if not settings.anthropic_api_key:
        print(f"  {FAIL}  ANTHROPIC_API_KEY is not set.")
        print(f"        {DIM}export ANTHROPIC_API_KEY=sk-ant-...{RESET}")
        return False
    key = settings.anthropic_api_key
    print(f"  {OK}  key present ({key[:7]}...{key[-4:]})")
    print(f"  {OK}  model: {settings.llm_model}")
    print(
        f"  {OK}  batch size {settings.llm_batch_size}, "
        f"{settings.llm_max_attempts} attempts, "
        f"{settings.llm_timeout_seconds:.0f}s timeout"
    )
    return True


def check_schema() -> bool:
    header(2, "Generated tool schema")
    schema = _tool_schema()
    raw = json.dumps(schema)
    problems = []
    if "$ref" in raw or "$defs" in raw:
        problems.append("unresolved $ref/$defs -- enum inlining did not work")
    item = schema.get("properties", {}).get("analyses", {}).get("items", {})
    props = item.get("properties", {})
    for field in ("id", "sentiment", "category", "severity", "summary", "theme"):
        if field not in props:
            problems.append(f"missing field: {field}")
    if props.get("severity", {}).get("maximum") != 5:
        problems.append("severity bounds missing -- out-of-range values won't be constrained")
    if not props.get("sentiment", {}).get("enum"):
        problems.append("sentiment enum missing -- the model can return free text")

    for p in problems:
        print(f"  {FAIL}  {p}")
    if problems:
        return False
    print(f"  {OK}  self-contained, {len(props)} fields, enums and bounds present")
    print(f"  {DIM}      {len(raw)} bytes{RESET}")
    return True


async def check_api(settings: Settings) -> bool:
    header(3, "Live API call (1 item)")
    analyzer = LLMAnalyzer(settings)
    try:
        result = await analyzer.analyze(
            [FeedbackItem(id="probe", text="The export button throws a 500 error.")],
            "B2B analytics SaaS.",
        )
    except Exception as exc:
        print(f"  {FAIL}  {type(exc).__name__}: {str(exc)[:300]}")
        detail = getattr(exc, "detail", None)
        if detail:
            print(f"  {DIM}      {str(detail)[:300]}{RESET}")
        text = f"{exc} {getattr(exc, 'detail', '')}"
        if "401" in text or "403" in text:
            hint = "Bad or missing API key. Check ANTHROPIC_API_KEY."
        elif "model" in text.lower() and "not" in text.lower():
            hint = (
                "The model name is not available to this account. "
                "Set LLM_MODEL to an identifier from your console."
            )
        elif "tool_choice" in text:
            hint = (
                "This model does not support forced tool choice. The client "
                "negotiates down automatically; seeing this means the "
                "downgrade path failed. Try LLM_TOOL_CHOICE=auto."
            )
        elif "400" in text:
            hint = (
                "The request was rejected. If it names input_schema, print it: "
                "python -c 'from app.services.llm import _tool_schema; import json;"
                " print(json.dumps(_tool_schema(), indent=2))'"
            )
        else:
            hint = "Check connectivity and the message above."
        print(f"\n  {DIM}{hint}{RESET}")
        return False

    if not result.analyses:
        print(f"  {FAIL}  call succeeded but returned no analyses")
        return False
    a = result.analyses[0]
    mode = "forced" if analyzer._forced else "auto (negotiated down)"
    print(f"  {OK}  tool call accepted and parsed")
    print(f"  {DIM}      tool_choice: {mode}{RESET}")
    print(f"  {DIM}      {a.category.value} / {a.sentiment.value} / severity {a.severity}")
    print(f"        theme: {a.theme!r}")
    print(f"        {result.input_tokens} in + {result.output_tokens} out tokens{RESET}")
    return True


async def check_clustering(settings: Settings) -> bool:
    header(4, "Theme reuse (6 items, 2 real clusters)")
    analyzer = LLMAnalyzer(settings)
    try:
        result = await analyzer.analyze(
            [FeedbackItem(id=i, text=t) for i, t in PROBE],
            "B2B analytics SaaS. Users build dashboards and export reports.",
        )
    except Exception as exc:
        print(f"  {FAIL}  {type(exc).__name__}: {str(exc)[:200]}")
        return False

    themes = [a.theme.strip().lower() for a in result.analyses]
    counts = Counter(themes)
    for a in result.analyses:
        print(f"  {DIM}      {a.id}  sev{a.severity}  {a.theme!r}{RESET}")

    distinct = len(counts)
    print(
        f"\n  {DIM}      {distinct} distinct themes for 6 items "
        f"({result.input_tokens + result.output_tokens} tokens){RESET}"
    )

    if distinct == 2:
        print(f"  {OK}  perfect clustering -- string-keyed aggregation will work as designed")
        return True
    if distinct <= 3:
        print(
            f"  {WARN}  close but not exact. Aggregation will mostly work; "
            f"expect some split themes."
        )
        print(
            f"  {DIM}      Strengthen the theme-reuse rule in SYSTEM_PROMPT, "
            f"then re-run.{RESET}"
        )
        return True
    if distinct >= 5:
        print(f"  {FAIL}  the model is not reusing theme wording.")
        print(f"  {DIM}      Nothing will crash, but every item becomes its own 'theme',")
        print("        producing near-duplicate GitHub issues and a useless digest.")
        print("        Fix SYSTEM_PROMPT before running the full eval, or move to")
        print(f"        embedding-based clustering.{RESET}")
        return False
    print(f"  {WARN}  {distinct} themes -- partial clustering. Check the eval's theme_ari.")
    return True


def next_steps() -> None:
    header(5, "Next")
    print(
        f"  {DIM}1.{RESET} make eval                  "
        f"{DIM}# score the 50-item golden set, 3 repeats{RESET}"
    )
    print(
        f"  {DIM}2.{RESET} tune thresholds to measured-minus-margin in evals/thresholds.json"
    )
    print(
        f"  {DIM}3.{RESET} make eval-baseline         {DIM}# commit the live baseline{RESET}"
    )
    print(
        f"  {DIM}4.{RESET} set GITHUB_TOKEN / GITHUB_REPO / SLACK_WEBHOOK_URL, "
        f"then POST samples/request.json with dry_run=false"
    )


async def main() -> int:
    settings = Settings()
    print(f"\n{DIM}FeedbackIQ preflight{RESET}")

    if not check_config(settings):
        return 2
    if not check_schema():
        return 1
    if not await check_api(settings):
        return 1
    if not await check_clustering(settings):
        return 1

    next_steps()
    print(f"\n{GREEN}Ready for the live eval.{RESET}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
