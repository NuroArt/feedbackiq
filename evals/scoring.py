"""Scoring metrics for the golden-set eval.

Pure functions over (gold, predicted) label lists. No I/O, no model, no config
-- so every metric here is unit-tested against hand-computed values in
`tests/test_evals.py`. A scorer you can't test is a scorer you can't trust to
gate a deploy.

Metric choices, and why each one is here:

* **accuracy** -- the headline number, but it hides minority-class collapse.
* **macro-F1** -- averages F1 per class with equal weight, so `docs` (4 items)
  counts as much as `bug` (13). A model that stopped predicting `docs` entirely
  would barely dent accuracy and would visibly crater macro-F1. This is the
  metric that catches the regression that matters.
* **severity MAE + within-1** -- severity is ordinal, not categorical. Predicting
  4 when gold is 5 is nearly right; predicting 1 is not. Exact-match accuracy
  treats those the same, so it's the wrong tool. `within_1` is the operational
  question: would this have been routed to the correct urgency bucket?
* **Adjusted Rand Index** -- scores the *clustering*, comparing the partition
  the model's free-text themes induce against the gold partition. Chance-
  corrected, so 0.0 means "no better than random grouping" rather than some
  arbitrary floor. This is the only honest way to score a stage whose output
  vocabulary we don't control.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from math import comb


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def accuracy(gold: list[str], pred: list[str]) -> float:
    if not gold:
        return 0.0
    return sum(g == p for g, p in zip(gold, pred, strict=True)) / len(gold)


def per_class_f1(gold: list[str], pred: list[str]) -> dict[str, dict[str, float]]:
    """Precision / recall / F1 / support for every class present in either list."""
    classes = sorted(set(gold) | set(pred))
    out: dict[str, dict[str, float]] = {}
    for c in classes:
        tp = sum(g == c and p == c for g, p in zip(gold, pred, strict=True))
        fp = sum(g != c and p == c for g, p in zip(gold, pred, strict=True))
        fn = sum(g == c and p != c for g, p in zip(gold, pred, strict=True))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[c] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": gold.count(c),
        }
    return out


def macro_f1(gold: list[str], pred: list[str]) -> float:
    """Mean F1 over classes that appear in the gold labels.

    Classes the model hallucinated but that aren't in gold are excluded from the
    average -- otherwise inventing a junk label would *raise* the denominator
    and quietly lower the score for an unrelated reason.
    """
    scores = per_class_f1(gold, pred)
    gold_classes = set(gold)
    if not gold_classes:
        return 0.0
    return round(sum(scores[c]["f1"] for c in gold_classes) / len(gold_classes), 4)


def confusion_pairs(gold: list[str], pred: list[str], top: int = 5) -> list[dict]:
    """The most common (gold -> predicted) mistakes. Turns a bad score into a
    diagnosis: 'usability is being read as bug 6 times' is actionable."""
    wrong = Counter((g, p) for g, p in zip(gold, pred, strict=True) if g != p)
    return [{"gold": g, "predicted": p, "count": n} for (g, p), n in wrong.most_common(top)]


# ---------------------------------------------------------------------------
# Ordinal
# ---------------------------------------------------------------------------
def severity_scores(gold: list[int], pred: list[int]) -> dict[str, float]:
    if not gold:
        return {"mae": 0.0, "within_1": 0.0, "exact": 0.0, "bias": 0.0}
    diffs = [p - g for g, p in zip(gold, pred, strict=True)]
    n = len(diffs)
    return {
        "mae": round(sum(abs(d) for d in diffs) / n, 4),
        "within_1": round(sum(abs(d) <= 1 for d in diffs) / n, 4),
        "exact": round(sum(d == 0 for d in diffs) / n, 4),
        # Signed mean: positive = the model systematically over-escalates, which
        # floods the issue tracker; negative = it under-calls real incidents.
        "bias": round(sum(diffs) / n, 4),
    }


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def adjusted_rand_index(gold: list[str], pred: list[str]) -> float:
    """ARI between two partitions of the same items.

    1.0 = identical grouping, 0.0 = what random labelling would score,
    negative = worse than chance. Implemented directly rather than pulling in
    scikit-learn, which would add ~100MB to the image for one function.
    """
    n = len(gold)
    if n < 2:
        return 0.0

    contingency = Counter(zip(gold, pred, strict=True))
    gold_sizes = Counter(gold)
    pred_sizes = Counter(pred)

    index = sum(comb(v, 2) for v in contingency.values())
    exp_a = sum(comb(v, 2) for v in gold_sizes.values())
    exp_b = sum(comb(v, 2) for v in pred_sizes.values())
    total = comb(n, 2)

    expected = exp_a * exp_b / total
    maximum = (exp_a + exp_b) / 2
    if maximum == expected:
        return 1.0 if index == expected else 0.0
    return round((index - expected) / (maximum - expected), 4)


def cluster_shape(gold: list[str], pred: list[str]) -> dict[str, float]:
    """Over- and under-clustering, which ARI alone doesn't make legible.

    `cluster_ratio` > 1 means the model split gold themes apart (the tracker
    fills with near-duplicate issues); < 1 means it merged distinct problems
    into one issue, which is worse -- a real incident gets buried.
    """
    g, p = len(set(gold)), len(set(pred))
    # Pairs that gold says belong together and the model also grouped together.
    same_gold = [(a, b) for a, b in combinations(range(len(gold)), 2) if gold[a] == gold[b]]
    kept = sum(1 for a, b in same_gold if pred[a] == pred[b])
    return {
        "gold_clusters": g,
        "predicted_clusters": p,
        "cluster_ratio": round(p / g, 3) if g else 0.0,
        # Recall over co-membership: of the item pairs that should share a
        # theme, how many actually do.
        "pair_recall": round(kept / len(same_gold), 4) if same_gold else 0.0,
    }


# ---------------------------------------------------------------------------
# Aggregation over repeats
# ---------------------------------------------------------------------------
@dataclass
class MetricStat:
    """A metric measured across N repeats. Mean plus spread.

    Reporting a single run of a stochastic model as 'the score' is the classic
    eval mistake: the next run moves 3 points and you can't tell a regression
    from noise. `spread` is what the regression gate is tolerant of.
    """

    mean: float
    min: float
    max: float
    runs: list[float] = field(default_factory=list)

    @property
    def spread(self) -> float:
        return round(self.max - self.min, 4)

    def to_dict(self) -> dict:
        return {
            "mean": round(self.mean, 4),
            "min": round(self.min, 4),
            "max": round(self.max, 4),
            "spread": self.spread,
            "runs": [round(r, 4) for r in self.runs],
        }


def aggregate(values: list[float]) -> MetricStat:
    return MetricStat(
        mean=sum(values) / len(values), min=min(values), max=max(values), runs=values
    )
