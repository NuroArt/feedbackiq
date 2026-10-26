"""Golden-set eval runner.

    python -m evals.runner                      # one pass, current profile
    python -m evals.runner --repeats 3          # measure variance
    python -m evals.runner --update-baseline    # accept the current scores
    python -m evals.runner --profile stub       # free, deterministic, no key

Exit codes are the contract CI relies on:
    0  all thresholds met and no regression against the baseline
    1  a threshold was missed, or a metric dropped below
       baseline - regression_tolerance
    2  the run could not complete (no key, load failure)

Two gates, not one. **Thresholds** are the absolute floor: ship nothing worse
than this. **Baseline** catches the slow slide -- a change that stays above the
floor but is measurably worse than yesterday still fails, because that is how
quality erodes in practice, three points at a time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.config import Settings
from app.models import FeedbackItem
from app.services.llm import StubAnalyzer, build_analyzer
from evals.scoring import (
    accuracy,
    adjusted_rand_index,
    aggregate,
    cluster_shape,
    confusion_pairs,
    macro_f1,
    per_class_f1,
    severity_scores,
)

EVAL_DIR = Path(__file__).parent
GOLDEN_SET = EVAL_DIR / "golden_set.jsonl"
THRESHOLDS = EVAL_DIR / "thresholds.json"
BASELINE = EVAL_DIR / "baseline.json"

PRODUCT_CONTEXT = (
    "B2B analytics SaaS. Users build dashboards, schedule reports, import CSVs, "
    "and export data. Sold on per-seat plans with SSO on enterprise tiers."
)

# Metrics where a smaller number is better, so the comparison flips.
LOWER_IS_BETTER = {"severity_mae"}


@dataclass
class GoldItem:
    item: FeedbackItem
    sentiment: str
    category: str
    severity: int
    theme_id: str


def load_golden_set(path: Path = GOLDEN_SET, limit: int | None = None) -> list[GoldItem]:
    rows: list[GoldItem] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            raw = json.loads(line)
            gold = raw["gold"]
            rows.append(
                GoldItem(
                    item=FeedbackItem(
                        id=raw["id"],
                        text=raw["text"],
                        source=raw.get("source", "golden_set"),
                        customer_tier=raw.get("customer_tier", "unknown"),
                    ),
                    sentiment=gold["sentiment"],
                    category=gold["category"],
                    severity=int(gold["severity"]),
                    theme_id=gold["theme_id"],
                )
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            raise SystemExit(f"{path}:{line_no}: malformed golden item -- {exc}") from exc
    return rows[:limit] if limit else rows


# ---------------------------------------------------------------------------
async def score_once(gold_items: list[GoldItem], settings: Settings) -> dict:
    """One full pass over the golden set."""
    analyzer = build_analyzer(settings)
    started = time.perf_counter()
    result = await analyzer.analyze(
        [g.item for g in gold_items], product_context=PRODUCT_CONTEXT
    )
    elapsed = time.perf_counter() - started

    by_id = {a.id: a for a in result.analyses}
    # Score only items the model returned, and report coverage separately --
    # otherwise a model that silently drops hard items looks more accurate.
    scored = [g for g in gold_items if g.item.id in by_id]

    g_sent = [g.sentiment for g in scored]
    p_sent = [by_id[g.item.id].sentiment.value for g in scored]
    g_cat = [g.category for g in scored]
    p_cat = [by_id[g.item.id].category.value for g in scored]
    g_sev = [g.severity for g in scored]
    p_sev = [by_id[g.item.id].severity for g in scored]
    g_theme = [g.theme_id for g in scored]
    # Normalise the model's free-text theme the same way the pipeline does, so
    # the eval measures the clustering the product actually gets.
    p_theme = [by_id[g.item.id].theme.strip().lower() for g in scored]

    sev = severity_scores(g_sev, p_sev)
    return {
        "metrics": {
            "coverage": round(len(scored) / len(gold_items), 4),
            "sentiment_accuracy": round(accuracy(g_sent, p_sent), 4),
            "category_accuracy": round(accuracy(g_cat, p_cat), 4),
            "category_macro_f1": macro_f1(g_cat, p_cat),
            "severity_mae": sev["mae"],
            "severity_within_1": sev["within_1"],
            "severity_exact": sev["exact"],
            "severity_bias": sev["bias"],
            "theme_ari": adjusted_rand_index(g_theme, p_theme),
        },
        "diagnostics": {
            "category_per_class": per_class_f1(g_cat, p_cat),
            "category_confusions": confusion_pairs(g_cat, p_cat),
            "sentiment_confusions": confusion_pairs(g_sent, p_sent),
            "clustering": cluster_shape(g_theme, p_theme),
            "missing_ids": [g.item.id for g in gold_items if g.item.id not in by_id],
        },
        "cost": {
            "llm_calls": result.calls,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "duration_seconds": round(elapsed, 2),
        },
    }


async def run_eval(
    settings: Settings, repeats: int, limit: int | None = None
) -> dict:
    gold_items = load_golden_set(limit=limit)
    passes = [await score_once(gold_items, settings) for _ in range(repeats)]

    keys = passes[0]["metrics"].keys()
    aggregated = {
        k: aggregate([p["metrics"][k] for p in passes]).to_dict() for k in keys
    }
    stub = isinstance(build_analyzer(settings), StubAnalyzer)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "profile": "stub" if stub else "live",
        "model": "stub" if stub else settings.llm_model,
        "items": len(gold_items),
        "repeats": repeats,
        "metrics": aggregated,
        # Diagnostics from the final pass; the aggregate is the headline, the
        # confusion matrix is what you read when it fails.
        "diagnostics": passes[-1]["diagnostics"],
        "cost": {
            "llm_calls": sum(p["cost"]["llm_calls"] for p in passes),
            "input_tokens": sum(p["cost"]["input_tokens"] for p in passes),
            "output_tokens": sum(p["cost"]["output_tokens"] for p in passes),
            "duration_seconds": round(
                sum(p["cost"]["duration_seconds"] for p in passes), 2
            ),
        },
    }


# ---------------------------------------------------------------------------
def evaluate_gates(report: dict, thresholds: dict, baseline: dict | None) -> list[dict]:
    """Compare every gated metric against the floor and against the baseline."""
    profile = thresholds[report["profile"]]
    default_tolerance = profile.get("regression_tolerance", 0.0)
    # Per-metric tolerance: metrics vary by different amounts between runs, and
    # one global number either fails the noisy metric on noise or lets the
    # stable ones drift. Keys starting with "_" are documentation.
    overrides = {
        k: v
        for k, v in profile.get("regression_tolerance_overrides", {}).items()
        if not k.startswith("_")
    }
    findings: list[dict] = []

    for name, floor in profile.items():
        if name.startswith("_") or name not in report["metrics"]:
            continue
        if name in ("regression_tolerance", "regression_tolerance_overrides"):
            continue
        tolerance = overrides.get(name, default_tolerance)
        value = report["metrics"][name]["mean"]
        lower_better = name in LOWER_IS_BETTER

        meets_floor = value <= floor if lower_better else value >= floor
        finding = {
            "metric": name,
            "value": round(value, 4),
            "threshold": floor,
            "status": "pass" if meets_floor else "fail",
            "reason": "" if meets_floor else "below threshold",
        }

        if baseline and name in baseline.get("metrics", {}):
            prior = baseline["metrics"][name]["mean"]
            delta = value - prior
            finding["baseline"] = round(prior, 4)
            finding["delta"] = round(delta, 4)
            regressed = (
                delta > tolerance if lower_better else delta < -tolerance
            )
            if regressed:
                finding["status"] = "fail"
                finding["reason"] = (
                    f"regressed {abs(delta):.3f} vs baseline "
                    f"(tolerance {tolerance})"
                )
        findings.append(finding)
    return findings


def render_markdown(report: dict, findings: list[dict]) -> str:
    failed = [f for f in findings if f["status"] == "fail"]
    verdict = "**FAIL**" if failed else "**PASS**"
    cost = report["cost"]

    lines = [
        f"## Golden-set eval — {verdict}",
        "",
        f"`{report['model']}` · {report['items']} items · "
        f"{report['repeats']} repeat(s) · {cost['duration_seconds']}s · "
        f"{cost['input_tokens'] + cost['output_tokens']} tokens",
        "",
        "| Metric | Score | Spread | Threshold | Baseline | Δ | |",
        "| --- | ---: | ---: | ---: | ---: | ---: | :-: |",
    ]
    for f in findings:
        stat = report["metrics"][f["metric"]]
        mark = "✅" if f["status"] == "pass" else "❌"
        base = f.get("baseline", "—")
        delta = f.get("delta")
        delta_s = f"{delta:+.3f}" if delta is not None else "—"
        lines.append(
            f"| `{f['metric']}` | {f['value']:.3f} | ±{stat['spread']:.3f} | "
            f"{f['threshold']} | {base} | {delta_s} | {mark} |"
        )

    if failed:
        lines += ["", "### Failures", ""]
        lines += [
            f"- `{f['metric']}` {f['value']:.3f} — {f['reason']}" for f in failed
        ]

    confusions = report["diagnostics"]["category_confusions"]
    if confusions:
        lines += ["", "### Top category confusions", ""]
        lines += [
            f"- `{c['gold']}` read as `{c['predicted']}` x{c['count']}"
            for c in confusions
        ]

    weak = sorted(
        report["diagnostics"]["category_per_class"].items(),
        key=lambda kv: kv[1]["f1"],
    )[:3]
    lines += ["", "### Weakest classes", ""]
    lines += [
        f"- `{name}` F1 {s['f1']:.2f} (precision {s['precision']:.2f}, "
        f"recall {s['recall']:.2f}, n={s['support']})"
        for name, s in weak
    ]

    cl = report["diagnostics"]["clustering"]
    lines += [
        "",
        f"Clustering: {cl['predicted_clusters']} themes predicted vs "
        f"{cl['gold_clusters']} gold (ratio {cl['cluster_ratio']}), "
        f"pair recall {cl['pair_recall']:.2f}.",
    ]
    if report["diagnostics"]["missing_ids"]:
        lines.append(
            f"Unreturned items: `{', '.join(report['diagnostics']['missing_ids'])}`"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the golden-set eval.")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Passes over the set; >1 measures run-to-run variance.")
    parser.add_argument("--profile", choices=["live", "stub"], default=None,
                        help="Default: live when a key is present, else stub.")
    parser.add_argument("--limit", type=int, default=None, help="Score the first N items.")
    parser.add_argument("--out", type=Path, default=Path("eval-report.json"))
    parser.add_argument("--markdown", type=Path, default=Path("eval-report.md"))
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--update-baseline", action="store_true",
                        help="Write this run's scores as the new baseline.")
    parser.add_argument("--no-gate", action="store_true",
                        help="Report only; always exit 0.")
    args = parser.parse_args(argv)

    settings = Settings(llm_offline_mode=args.profile == "stub")
    if args.profile == "live" and not settings.llm_enabled:
        print("ERROR: --profile live requires ANTHROPIC_API_KEY.", file=sys.stderr)
        return 2

    report = asyncio.run(run_eval(settings, args.repeats, args.limit))

    thresholds = json.loads(THRESHOLDS.read_text())
    baseline = (
        json.loads(args.baseline.read_text()) if args.baseline.exists() else None
    )
    # Only compare like with like: a stub baseline must not gate a live run.
    if baseline and baseline.get("profile") != report["profile"]:
        baseline = None

    findings = evaluate_gates(report, thresholds, baseline)
    report["findings"] = findings
    report["passed"] = all(f["status"] == "pass" for f in findings)

    args.out.write_text(json.dumps(report, indent=2) + "\n")
    markdown = render_markdown(report, findings)
    args.markdown.write_text(markdown + "\n")
    print(markdown)

    if args.update_baseline:
        args.baseline.write_text(
            json.dumps(
                {
                    "generated_at": report["generated_at"],
                    "profile": report["profile"],
                    "model": report["model"],
                    "repeats": report["repeats"],
                    "metrics": report["metrics"],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nBaseline updated: {args.baseline}")

    if args.no_gate:
        return 0
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
