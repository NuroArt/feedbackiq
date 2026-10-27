"""Tests for the eval harness itself.

A scorer that gates deploys has to be trustworthy, so every metric is checked
against a hand-computed value rather than a snapshot of its own output. These
run in stub mode: hermetic, free, no key.
"""

from __future__ import annotations

import json

import pytest

from evals.runner import (
    GOLDEN_SET,
    evaluate_gates,
    load_golden_set,
    render_markdown,
    run_eval,
)
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


# ---------------------------------------------------------------------------
# Metric correctness
# ---------------------------------------------------------------------------
def test_accuracy_is_plain_agreement():
    assert accuracy(["a", "b", "c", "d"], ["a", "b", "c", "x"]) == 0.75


def test_macro_f1_punishes_minority_class_collapse():
    """The reason macro-F1 is the gated metric and accuracy is not."""
    gold = ["bug"] * 18 + ["docs"] * 2
    # Stops predicting `docs` entirely.
    pred = ["bug"] * 20
    assert accuracy(gold, pred) == 0.9  # looks fine
    assert macro_f1(gold, pred) < 0.5  # is not fine


def test_macro_f1_ignores_hallucinated_classes():
    """A label the model invented must not dilute the average."""
    gold = ["bug", "bug", "docs", "docs"]
    clean = ["bug", "bug", "docs", "docs"]
    noisy = ["bug", "bug", "docs", "invented"]
    assert macro_f1(gold, clean) == 1.0
    # The invented label costs recall on `docs`, but adds no third class.
    assert 0.0 < macro_f1(gold, noisy) < 1.0


def test_per_class_f1_reports_support_from_gold():
    scores = per_class_f1(["bug", "bug", "docs"], ["bug", "docs", "docs"])
    assert scores["bug"]["support"] == 2
    assert scores["bug"]["precision"] == 1.0  # 1 predicted bug, correct
    assert scores["bug"]["recall"] == 0.5  # 1 of 2 found


def test_severity_treats_near_misses_as_near():
    gold = [5, 5, 3, 1]
    pred = [4, 5, 3, 1]
    s = severity_scores(gold, pred)
    assert s["mae"] == 0.25
    assert s["within_1"] == 1.0
    assert s["exact"] == 0.75


def test_severity_bias_detects_systematic_over_escalation():
    s = severity_scores([2, 2, 2, 2], [4, 4, 4, 4])
    assert s["bias"] == 2.0  # positive = floods the issue tracker
    s2 = severity_scores([4, 4], [2, 2])
    assert s2["bias"] == -2.0  # negative = misses real incidents


def test_ari_is_one_for_identical_partitions_regardless_of_naming():
    gold = ["export", "export", "slow", "slow"]
    # Same grouping, completely different vocabulary.
    pred = ["cluster_7", "cluster_7", "zzz", "zzz"]
    assert adjusted_rand_index(gold, pred) == 1.0


def test_ari_penalises_merging_distinct_problems():
    gold = ["a", "a", "b", "b"]
    everything_one_bucket = ["x", "x", "x", "x"]
    assert adjusted_rand_index(gold, everything_one_bucket) <= 0.0


def test_ari_penalises_splitting_one_problem():
    gold = ["a", "a", "a", "a"]
    all_separate = ["1", "2", "3", "4"]
    assert adjusted_rand_index(gold, all_separate) <= 0.0


def test_cluster_shape_names_the_failure_direction():
    gold = ["a", "a", "b", "b"]
    over = cluster_shape(gold, ["1", "2", "3", "4"])
    assert over["cluster_ratio"] > 1 and over["pair_recall"] == 0.0
    under = cluster_shape(gold, ["x", "x", "x", "x"])
    assert under["cluster_ratio"] < 1 and under["pair_recall"] == 1.0


def test_confusion_pairs_rank_the_worst_mistake_first():
    gold = ["usability"] * 5 + ["bug"]
    pred = ["bug"] * 5 + ["bug"]
    top = confusion_pairs(gold, pred)[0]
    assert top == {"gold": "usability", "predicted": "bug", "count": 5}


def test_aggregate_reports_spread_across_repeats():
    stat = aggregate([0.80, 0.84, 0.88])
    assert stat.to_dict()["mean"] == 0.84
    assert stat.spread == pytest.approx(0.08)


# ---------------------------------------------------------------------------
# Golden set integrity -- the data is part of the contract
# ---------------------------------------------------------------------------
def test_golden_set_loads_and_is_the_expected_size():
    items = load_golden_set()
    assert len(items) == 50
    assert len({i.item.id for i in items}) == 50, "duplicate ids"


def test_golden_labels_are_all_valid_enum_members():
    from app.models import Category, Sentiment

    for g in load_golden_set():
        assert g.sentiment in {s.value for s in Sentiment}
        assert g.category in {c.value for c in Category}
        assert 1 <= g.severity <= 5


def test_every_gold_theme_has_at_least_three_items():
    """A cluster of one makes ARI noisy and macro-F1 meaningless."""
    from collections import Counter

    counts = Counter(g.theme_id for g in load_golden_set())
    assert all(n >= 3 for n in counts.values()), counts


def test_golden_set_covers_both_ends_of_the_severity_scale():
    severities = {g.severity for g in load_golden_set()}
    assert 1 in severities and 5 in severities


def test_golden_set_is_not_all_negative():
    """Guards against a set that would reward a model biased toward doom."""
    sentiments = [g.sentiment for g in load_golden_set()]
    assert sentiments.count("positive") >= 5
    assert sentiments.count("neutral") >= 5


# ---------------------------------------------------------------------------
# Gating logic -- the part that actually fails a build
# ---------------------------------------------------------------------------
def _report(**metrics) -> dict:
    return {
        "profile": "live",
        "metrics": {k: {"mean": v, "spread": 0.0} for k, v in metrics.items()},
        "diagnostics": {
            "category_confusions": [],
            "category_per_class": {
                "bug": {"f1": 0.9, "precision": 0.9, "recall": 0.9, "support": 10}
            },
            "clustering": {
                "gold_clusters": 12,
                "predicted_clusters": 12,
                "cluster_ratio": 1.0,
                "pair_recall": 0.9,
            },
            "missing_ids": [],
        },
        "cost": {"duration_seconds": 1.0, "input_tokens": 10, "output_tokens": 5},
        "model": "test",
        "items": 50,
        "repeats": 1,
    }


THRESHOLDS = {
    "live": {"category_macro_f1": 0.75, "severity_mae": 0.75, "regression_tolerance": 0.03}
}


def test_metric_below_floor_fails():
    findings = evaluate_gates(
        _report(category_macro_f1=0.70, severity_mae=0.5), THRESHOLDS, None
    )
    f1 = next(f for f in findings if f["metric"] == "category_macro_f1")
    assert f1["status"] == "fail" and f1["reason"] == "below threshold"


def test_lower_is_better_metric_inverts_the_comparison():
    """MAE of 0.9 is worse than a 0.75 floor even though the number is bigger."""
    findings = evaluate_gates(
        _report(category_macro_f1=0.9, severity_mae=0.9), THRESHOLDS, None
    )
    mae = next(f for f in findings if f["metric"] == "severity_mae")
    assert mae["status"] == "fail"

    findings_ok = evaluate_gates(
        _report(category_macro_f1=0.9, severity_mae=0.4), THRESHOLDS, None
    )
    assert all(f["status"] == "pass" for f in findings_ok)


def test_regression_against_baseline_fails_even_when_above_the_floor():
    """The slow-slide gate: 0.82 clears the 0.75 floor but is 0.08 down."""
    baseline = {
        "profile": "live",
        "metrics": {"category_macro_f1": {"mean": 0.90}, "severity_mae": {"mean": 0.50}},
    }
    findings = evaluate_gates(
        _report(category_macro_f1=0.82, severity_mae=0.50), THRESHOLDS, baseline
    )
    f1 = next(f for f in findings if f["metric"] == "category_macro_f1")
    assert f1["status"] == "fail"
    assert "regressed" in f1["reason"]
    assert f1["delta"] == pytest.approx(-0.08)


def test_drift_within_tolerance_passes():
    """Run-to-run noise must not fail a build, or the gate gets ignored."""
    baseline = {
        "profile": "live",
        "metrics": {"category_macro_f1": {"mean": 0.90}, "severity_mae": {"mean": 0.50}},
    }
    findings = evaluate_gates(
        _report(category_macro_f1=0.88, severity_mae=0.52), THRESHOLDS, baseline
    )
    assert all(f["status"] == "pass" for f in findings)


def test_improvement_never_fails():
    baseline = {
        "profile": "live",
        "metrics": {"category_macro_f1": {"mean": 0.80}, "severity_mae": {"mean": 0.60}},
    }
    findings = evaluate_gates(
        _report(category_macro_f1=0.95, severity_mae=0.20), THRESHOLDS, baseline
    )
    assert all(f["status"] == "pass" for f in findings)
    assert next(f for f in findings if f["metric"] == "severity_mae")["delta"] < 0


# ---------------------------------------------------------------------------
# End to end, in stub mode
# ---------------------------------------------------------------------------
async def test_full_eval_run_produces_a_complete_report(settings):
    report = await run_eval(settings, repeats=2, limit=20)
    assert report["profile"] == "stub"
    assert report["items"] == 20
    assert report["metrics"]["coverage"]["mean"] == 1.0
    for key in ("sentiment_accuracy", "category_macro_f1", "severity_mae", "theme_ari"):
        assert key in report["metrics"]
    assert report["diagnostics"]["category_per_class"]


async def test_markdown_report_names_the_failure(settings):
    report = await run_eval(settings, repeats=1, limit=20)
    impossible = {"stub": {"category_macro_f1": 0.99, "regression_tolerance": 0.0}}
    findings = evaluate_gates(report, impossible, None)
    md = render_markdown(report, findings)
    assert "FAIL" in md and "category_macro_f1" in md


def test_thresholds_file_covers_every_gated_metric():
    """Catches a metric renamed in the runner but not in the thresholds."""
    thresholds = json.loads((GOLDEN_SET.parent / "thresholds.json").read_text())
    for profile in ("live", "stub"):
        keys = {
            k
            for k in thresholds[profile]
            if not k.startswith("_")
            and k not in {"regression_tolerance", "regression_tolerance_overrides"}
        }
        assert keys == {
            "sentiment_accuracy",
            "category_accuracy",
            "category_macro_f1",
            "severity_within_1",
            "severity_mae",
            "theme_ari",
            "coverage",
        }


# ---------------------------------------------------------------------------
# Per-metric regression tolerance
#
# Added after the first live run: severity_mae moved ±0.14 across three
# repeats while every accuracy metric was flat at ±0.00. One global tolerance
# either fails the noisy metric on noise or lets the stable ones drift.
# ---------------------------------------------------------------------------
TOLERANCE_PROFILE = {
    "live": {
        "category_macro_f1": 0.75,
        "severity_mae": 0.75,
        "regression_tolerance": 0.03,
        "regression_tolerance_overrides": {
            "_why": "documentation, must be ignored",
            "severity_mae": 0.3,
        },
    }
}


def test_override_lets_a_noisy_metric_drift_within_its_own_tolerance():
    baseline = {
        "profile": "live",
        "metrics": {"category_macro_f1": {"mean": 0.98}, "severity_mae": {"mean": 0.28}},
    }
    # +0.14 on MAE is one observed spread -- noise, not a regression.
    findings = evaluate_gates(
        _report(category_macro_f1=0.98, severity_mae=0.42),
        TOLERANCE_PROFILE,
        baseline,
    )
    assert all(f["status"] == "pass" for f in findings)


def test_override_still_catches_a_real_regression():
    baseline = {
        "profile": "live",
        "metrics": {"category_macro_f1": {"mean": 0.98}, "severity_mae": {"mean": 0.28}},
    }
    findings = evaluate_gates(
        _report(category_macro_f1=0.98, severity_mae=0.70),  # +0.42, past 0.3
        TOLERANCE_PROFILE,
        baseline,
    )
    mae = next(f for f in findings if f["metric"] == "severity_mae")
    assert mae["status"] == "fail" and "regressed" in mae["reason"]


def test_metrics_without_an_override_keep_the_global_tolerance():
    baseline = {
        "profile": "live",
        "metrics": {"category_macro_f1": {"mean": 0.98}, "severity_mae": {"mean": 0.28}},
    }
    findings = evaluate_gates(
        _report(category_macro_f1=0.90, severity_mae=0.28),  # -0.08, past 0.03
        TOLERANCE_PROFILE,
        baseline,
    )
    f1 = next(f for f in findings if f["metric"] == "category_macro_f1")
    assert f1["status"] == "fail"


def test_documentation_keys_are_not_treated_as_metrics():
    """Keys starting with _ carry provenance, not thresholds."""
    findings = evaluate_gates(
        _report(category_macro_f1=0.98, severity_mae=0.28),
        TOLERANCE_PROFILE,
        None,
    )
    assert {f["metric"] for f in findings} == {"category_macro_f1", "severity_mae"}


def test_live_thresholds_sit_below_the_first_measured_run():
    """A floor above measured fails on day one; far below never fires."""
    thresholds = json.loads((GOLDEN_SET.parent / "thresholds.json").read_text())
    live = thresholds["live"]
    measured = live["_measured_2026_10_25"]
    for metric, score in measured.items():
        if metric.startswith("_") or metric not in live:
            continue
        floor = live[metric]
        if metric == "severity_mae":
            assert floor > score, f"{metric}: ceiling {floor} must exceed measured {score}"
        else:
            assert floor <= score, f"{metric}: floor {floor} above measured {score}"
            assert floor >= score - 0.1, f"{metric}: floor {floor} too slack vs {score}"
