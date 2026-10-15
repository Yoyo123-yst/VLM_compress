# 5786 unknown-capacity real probe

## Scope

This is a development-only diagnostic, not a method result. The target was the
semantically valid complete wrong number `670` for the clean FastV answer
`7,120`. The candidate generator reported `estimated_feasible_at_eps=false`
only because its local Full-capacity proxy was unavailable (`full_clean_hard`
contained a nonpositive value).

## Protocol

- one fresh attack process on GPU 0;
- v30 dataset loader and clean admission;
- complete-target objective;
- ratio `0.25`, uint8 L-infinity budget `8`;
- at most 5 attempts and 5 accepted updates;
- every update quantized to PNG, reloaded, checked by Full exact/token exact,
  Full replay, FastV replay, and pixel budget;
- final PNG evaluated by a second fresh model process.

## Result

- accepted updates: **5/5**;
- attempts: **5/5**;
- nominal complete-sequence backward calls: **10**;
- Full exact/token exact: **pass** for every accepted update and independent
  final evaluation;
- final PNG uint8 L-infinity: **5**;
- independent replay: **pass**;
- final FastV text: **`7,120`**;
- target `670` hit: **no**;
- final FastV ANLS against the dataset answer: **1.0**.

## Decision

`LOCAL_CAPACITY_ESTIMATE_FALSE_NEGATIVE_BUT_TARGET_NOT_REACHED`.

The old selector must not equate an unavailable local capacity estimate with a
real infeasibility proof. However, this five-step probe does not establish an
attack success or justify a full run. The candidate should remain labelled
`UNKNOWN_FULL_CAPACITY` and be handled by a bounded real probe. Do not count
this probe in the v30 comparative method table and do not open unseen data.

## Fixed extension

A separately stored extension used the same target, objective, ratio, pixel
budget, and guard for 10 accepted updates:

- accepted updates: **10/10**;
- final PNG uint8 L-infinity: **8**;
- Full exact/token exact and replay: **pass**;
- final FastV text: **`7,120`**;
- target `670` hit: **no**;
- FastV ANLS: **1.0**.

This meets the pre-registered stop condition:
`REAL_PROBE_NO_TARGET_HIT`. The 5786 branch is closed; no larger run or unseen
evaluation should be launched from it.

## Next gate

The next action is analysis-only: update the target-selection protocol so an
unavailable local estimate is recorded as `UNKNOWN_FULL_CAPACITY`, rather than
as an explicit infeasibility result. Do not use this failed target to claim
that all complete-target attacks are impossible.
