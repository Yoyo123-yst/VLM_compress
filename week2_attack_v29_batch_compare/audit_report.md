# v29 post-run audit

## What was audited

The audit used only the 45 independently re-evaluated final PNGs. It did not select an intermediate checkpoint and did not open the reserved test set.

## Cost-normalized result

| method | backward attempts | accepted updates | wall time (s) | target hits | ANLS<0.5 |
|---|---:|---:|---:|---:|---:|
| Native-FDP | 300 | 300 | 3878.8 | n/a | 7/15 |
| complete-target | 195 | 195 | 3061.0 | 7/15 | 6/15 |
| complete-front | 150 | 146 | 2655.3 | 6/15 | 6/15 |

The attempt counts differ because the protocol assigns 20, 13, and 10 attempts respectively; the nominal backward budget is equalized by 2, 3, and 4 backward calls per attempt, but the result files expose attempt-level counters rather than a common wall-time budget. Therefore the lower wall time of complete-front is not evidence of a better attack: it used fewer outer attempts and had four rejected updates.

## Semantic interpretation

Exact target matching is a directional-control metric. It is not by itself a content-failure metric. The conservative primary endpoint remains the natural FastV output after the fixed final selector, with Full exact and independent replay required. Under that endpoint, complete-target and complete-front do not exceed Native-FDP on this cohort.

## Decision

Freeze `complete-target` only as a reproducible targeted-attack baseline if a targeted baseline is needed. Do not promote `complete-front` to a method contribution. Do not launch unseen evaluation from v29. Any future method must first show an independently measured gain over both Native-FDP and complete-target on a newly frozen development cohort with valid-target coverage reported separately.
