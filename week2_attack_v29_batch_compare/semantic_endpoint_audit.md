# v29 question-specific semantic endpoint audit

## Rule frozen for this audit

`CONTENT_FAILURE` means that the final natural FastV answer gives a clear, answer-type-compatible but wrong entity, number, year, amount, Boolean value, location, product, or objective. `ANSWER_PRESERVED` includes harmless shortening or wording changes that retain the requested answer. `INVALID_OR_AMBIGUOUS` is used for a malformed or non-answer response, and is reported separately rather than promoted to a semantic failure. All entries below already satisfy Full text/token exact and independent PNG replay.

| id | dataset answer | Native-FDP | complete-target | complete-front |
|---|---|---|---|---|
| 55341 | 71.6 | 0.00225 — CONTENT_FAILURE | 6.25 — CONTENT_FAILURE, TARGET_HIT | 6.25 — CONTENT_FAILURE, TARGET_HIT |
| 55192 | No | No — ANSWER_PRESERVED | No — ANSWER_PRESERVED | No — ANSWER_PRESERVED |
| 63843 | 1000 | 1000 — ANSWER_PRESERVED | 1000 — ANSWER_PRESERVED | 1000 — ANSWER_PRESERVED |
| 56601 | 1985 | 1985 — ANSWER_PRESERVED | 1988 — CONTENT_FAILURE, TARGET_HIT | 1988 — CONTENT_FAILURE, TARGET_HIT |
| 57069 | 32 | 32 — ANSWER_PRESERVED | 32 — ANSWER_PRESERVED | 32 — ANSWER_PRESERVED |
| 4415 | Strong share growth | Kool target segmentation — CONTENT_FAILURE | Influence share growth where Kool is in selected neighborhoods — CONTENT_FAILURE, TARGET_HIT | Influence Share Growth — CONTENT_FAILURE |
| 65311 | Barranquilla | Barruqued — INVALID_OR_AMBIGUOUS | Barrqueros — INVALID_OR_AMBIGUOUS | Barquisimeto, Venezuela — CONTENT_FAILURE, TARGET_HIT |
| 59042 | 3000.00 | $10,000 — CONTENT_FAILURE | 1,000 — CONTENT_FAILURE, TARGET_HIT | $1,000 — CONTENT_FAILURE, TARGET_HIT |
| 58647 | cigarette smoking is dangerous to your health | tobacco products are dangerous to your health — ANSWER_PRESERVED | Smoking is dangerous to your health — ANSWER_PRESERVED | Smoking is dangerous to your health — ANSWER_PRESERVED |
| 5942 | 520 | 520 — ANSWER_PRESERVED | 52 — CONTENT_FAILURE | 52 — CONTENT_FAILURE |
| 50763 | No | no — ANSWER_PRESERVED | yes — CONTENT_FAILURE, TARGET_HIT | yes — CONTENT_FAILURE, TARGET_HIT |
| 4449 | Turkish Gold | CAMEL TURKISH FAMILY — CONTENT_FAILURE | Less tobacco flavor — INVALID_OR_AMBIGUOUS | Less tobacco flavor — INVALID_OR_AMBIGUOUS |
| 8560 | Fred L. Smith, Jr. | Marine J. J. — INVALID_OR_AMBIGUOUS | John L. Smith, Jr. — CONTENT_FAILURE, TARGET_HIT | John L. Smith, Jr. — CONTENT_FAILURE, TARGET_HIT |
| 51024 | Dinner | DEBUT — CONTENT_FAILURE | Dinner — ANSWER_PRESERVED | Dinner — ANSWER_PRESERVED |
| 63839 | 114% | 22% — CONTENT_FAILURE | 111.4% — CONTENT_FAILURE, TARGET_HIT | 111.1% — CONTENT_FAILURE |

## Semantic totals on the 15 valid-target development samples

| method | CONTENT_FAILURE | ANSWER_PRESERVED | INVALID_OR_AMBIGUOUS | exact target hit |
|---|---:|---:|---:|---:|
| Native-FDP | 6/15 | 6/15 | 3/15 | n/a |
| complete-target | 8/15 | 5/15 | 2/15 | 7/15 |
| complete-front | 9/15 | 5/15 | 1/15 | 6/15 |

## Interpretation boundary

These are development results after method selection and target review. They establish a signal worth validating, not a generalization result. The meaningful finding is that the front constraint reaches three additional clear semantic failures over Native-FDP and one over complete-target on this fixed cohort, despite a lower exact-target-hit count than complete-target. The gain is driven by 4415, 5942, 63839, and 65311; it must be tested on a newly frozen development cohort before the method is retained.
