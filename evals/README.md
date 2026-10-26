# Golden set

50 hand-labelled feedback items in `golden_set.jsonl`, used to put a number on
classification quality so prompt and model changes can be reviewed as a diff.

## Schema

```json
{"id": "g001", "text": "...", "source": "zendesk", "customer_tier": "enterprise",
 "gold": {"sentiment": "negative", "category": "bug", "severity": 5,
          "theme_id": "export_failure"}}
```

`theme_id` is a **cluster label, not a string the model is expected to emit.**
The model invents its own theme wording; the scorer compares the *partition*
those strings induce against the gold partition using the Adjusted Rand Index.
That measures "did it group the right items together" without demanding it
guess our vocabulary.

## Composition

| Theme | Items | Category | Why it's in the set |
| --- | --- | --- | --- |
| `export_failure` | 5 | bug | The high-severity case that must become a GitHub issue |
| `dashboard_slow` | 5 | performance | Tests bug/performance boundary |
| `csv_upload_crash` | 4 | bug | Data loss — severity ceiling |
| `sso_login_failure` | 4 | bug | Must not be clustered with the other bugs |
| `billing_settings_hidden` | 4 | usability | Low severity, must *not* page anyone |
| `mobile_layout_broken` | 3 | usability | Tests usability/bug boundary |
| `salesforce_integration` | 4 | feature_request | Neutral sentiment, non-trivial severity |
| `scheduled_reports` | 4 | feature_request | Must not merge with the Salesforce ask |
| `renewal_price_increase` | 4 | pricing | Emotionally loaded, low technical severity |
| `support_response_time` | 4 | support | Tests support/other boundary |
| `webhook_docs_gaps` | 4 | docs | Rare class — the one macro-F1 protects |
| `reporting_praise` | 5 | other | Positive baseline; guards against doom bias |

The set is deliberately adversarial in four places:

- **`g022`** ("evaluated a competitor because they sync with Salesforce") is a
  feature request with *negative* sentiment — sentiment and category must be
  judged independently.
- **`g023`–`g026`** are angry but operationally low-severity. A model that reads
  severity off emotional intensity scores badly here, which is the point.
- **`g044`–`g046`** (mobile layout) sit on the usability/bug line.
- **`g014`** ("importer silently drops rows") is severity 5 despite calm wording,
  because it is silent data loss.

## Labelling rules

- **Severity** is customer impact, not tone. 5 = cannot use the product or data
  loss. 4 = core workflow broken. 3 = significant friction with a workaround.
  2 = minor annoyance. 1 = cosmetic or praise.
- **Sentiment** is the author's attitude; a neutral feature request stays neutral.
- **Category** is the single best fit. Where two fit, the one a triage engineer
  would route on wins.

## Extending it

Add items when a real classification failure is found in production — the set
should grow out of incidents, not out of imagination. Keep classes roughly
balanced; macro-F1 will punish a class that shrinks to one or two items.
After adding, re-baseline deliberately:

```bash
python -m evals.runner --repeats 3 --update-baseline
```

and commit `baseline.json` in the same PR as the new items, so the diff shows
both the new data and its effect.
