# Target-freeze audit before the next comparison

Date: 2026-10-15
Ratio: 0.25

## Decision

`PARTIAL_TARGET_ADMISSION_60700_ONLY`

The next three-way training comparison must not start until 60700 and 39204
each have a pre-registered, complete, semantically valid wrong answer. The
existing two-target pilot remains valid for 47450 and 7006 and is not changed.

## 39204

The available `candidate_manifest_v2/39204.json` contains token substitutions
such as `FOOD TECHNOLOGY -> FOOD TECHNLOGY`, `TECHNOLOGY`, and malformed
continuations such as `FOOD TELECHNOLOGY`. These are spelling/truncation or
near-string variants, not a vetted complete wrong answer. They must not be
promoted to a target merely because their forced-continuation ANLS is low.

The fresh clean-only generator produced six candidates, but all are spelling,
truncation, or near-answer variants (`FOOD TECHNLOGY`, `TECHNOLOGY`,
`FOOD TELECHNOLOGY`, `FOOF TECHNOLOGY`, `FOOD TECHNOLOGICAL`, and
`FOody technology`). None is a semantically vetted complete wrong answer.

Status: `NO_VALID_PREREGISTERED_COMPLETE_WRONG_TARGET`.

A second clean-only pass used 20 fixed-seed diverse generations (top-p 0.92,
temperature 0.8). It produced only the clean answer `FOOD TECHNOLOGY` (one
unique output), confirming that this is not merely a narrow top-k branch-search
failure.

## 60700

The historical semantic manifest records `PPSB` as a possible content failure,
but the v20 report explicitly says that `PPSB` did not occur in the clean-start
formal trajectory. It is therefore an observed historical candidate, not a
pre-registered target for this comparison.

The fresh clean-only generator produced six candidates. Several are complete
non-alias phrases (for example `Page 5`, `Pages 3 and 4`, and `Dr. H. S.
SORBS`), but they still require an explicit pre-registered semantic choice
before optimization. `PPSB` remains excluded because it is an old observed
output, not a clean-start target.

Status: `CANDIDATES_AVAILABLE_PENDING_TARGET_FREEZE`.

## Required next action

Create a new frozen target manifest for both samples using the same clean-start
candidate protocol used for 47450 and 7006:

1. Generate at most three complete continuations from clean FastV.
2. Exclude aliases, spelling changes, truncations, punctuation-only changes,
   and malformed outputs.
3. Record token IDs, decoded text, candidate rank, manifest hash, and manual
   semantic status before any attack optimization.
4. If either sample has no valid target, keep it out of the targeted comparison
   and report it as blocked; do not invent a target from an old attack output.

Only after this manifest is frozen may the three-way run begin. At present this
permits a 60700-only comparison if one candidate is frozen by rule; 39204 must
remain blocked. The comparison is:
Native-FDP vs complete-target vs complete-target + first-divergence constraint.

## Time estimate after target freeze

Target generation/review: approximately 5--15 minutes, no GPU training.

Two samples x three methods x ten accepted updates: approximately 25--40
minutes on four GPUs, including independent replay and final four-ratio PNG
evaluation. If both samples require 20 accepted updates, allow approximately
50--80 minutes. These are estimates, not completed measurements.
