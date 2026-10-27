"""Theme matching for cross-run issue deduplication.

The problem this solves, found on the first live double-run:

    run 1: "Export fails with server error"
    run 2: "Report export fails with error"

Same underlying problem, different wording, so a hash of the theme name gives
two different fingerprints and the tracker fills with duplicates. The model is
not misbehaving -- it is generating prose, and prose varies.

Note carefully what the golden-set eval did and did not establish. `theme_ari`
of 0.966 measured whether items that belong together were *grouped* together
within a single run. It said nothing about whether the same cluster receives
the same *name* on a later run. Those are different properties, and only the
first was being measured.

The fix is to stop treating the theme string as an identifier and start
treating it as evidence. Matching runs in two stages:

1. **Exact fingerprint** -- a hash of category + normalized name. Free, and
   catches the common case where wording happens to be stable.
2. **Token overlap** -- Jaccard similarity over content words, requiring the
   same category. Catches re-wordings.

Deliberately not embeddings. This needs to be deterministic (the same inputs
must match tomorrow), explainable (an operator has to be able to see why two
themes were considered the same), and free of a network call in the matching
path. Jaccard over stemmed content words is all three. If it proves too blunt
in practice, that is a measurable failure and the eval can be extended to
catch it -- which is the point of having one.
"""

from __future__ import annotations

import hashlib
import re

# Words that carry no discriminating signal in a feedback theme. "Error",
# "fails" and "issue" are deliberately NOT here: they distinguish a bug theme
# from a feature request.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "there",
        "this",
        "to",
        "when",
        "with",
        "without",
    }
)

_WORD = re.compile(r"[a-z0-9]+")

# Crude suffix stripping. A real stemmer (Snowball) would be better, but it is
# a dependency for four rules, and these four cover the variation actually
# observed: crashes/crashing, loses/losing, fails/failing, loads/loading.
_SUFFIXES = ("ing", "ed", "es", "s")


def normalize_tokens(text: str) -> frozenset[str]:
    """Content words of a theme name, lowercased, destemmed, deduplicated."""
    tokens = set()
    for word in _WORD.findall(text.lower()):
        if word in _STOPWORDS or len(word) < 2:
            continue
        for suffix in _SUFFIXES:
            if len(word) > len(suffix) + 2 and word.endswith(suffix):
                word = word[: -len(suffix)]
                break
        tokens.add(word)
    return frozenset(tokens)


def canonical_key(category: str, name: str) -> str:
    """A fingerprint that survives word order and light re-wording.

    Sorting the normalized tokens means "export fails with server error" and
    "server error when export fails" produce the same key.
    """
    tokens = "|".join(sorted(normalize_tokens(name)))
    return hashlib.sha256(f"{category}|{tokens}".encode()).hexdigest()[:16]


def similarity(a: str, b: str) -> float:
    """Jaccard overlap of content words. 1.0 identical, 0.0 disjoint."""
    ta, tb = normalize_tokens(a), normalize_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# Tuned against the observed re-wordings:
#   "export fails with server error" vs "report export fails with error" -> 0.60
#   "csv upload crashes losing data" vs "csv upload crashes and loses data" -> 1.00
# and against pairs that must NOT match:
#   "slow dashboard and chart loading" vs "csv upload crashes losing data" -> 0.0
MATCH_THRESHOLD = 0.5


def best_match(
    category: str, name: str, candidates: list[tuple[str, str, str]]
) -> tuple[str, float] | None:
    """Pick the existing issue a theme belongs to.

    `candidates` is (issue_ref, candidate_category, candidate_name). Category
    must match exactly -- a usability theme and a bug theme with overlapping
    words are different problems, and merging them buries one of them.
    """
    best: tuple[str, float] | None = None
    for ref, cand_category, cand_name in candidates:
        if cand_category != category:
            continue
        score = similarity(name, cand_name)
        if score >= MATCH_THRESHOLD and (best is None or score > best[1]):
            best = (ref, score)
    return best
