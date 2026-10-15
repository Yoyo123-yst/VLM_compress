# Budget-conditioned protocol final report

## Decision

NO_GO_NO_COMPRESSION_CONDITIONAL_EFFECT

Stage I gate D failed. B+ was not implemented. No new PGD, development-set arm, or new pristine cohort was started.

## What this cohort is

The numbers below are a read-only diagnosis of the existing pristine60 Variant B images and the matched single-noise PNGs. They are not a B+ result and they are not a new unseen test.

## FDP

CPU tests passed. GPU health on historical sample 1025 passed: the loss is finite, EOS is included, teacher-forced argmax matches the clean tokens, the evidence-page gradient is 0, and the auxiliary-page gradient is finite and nonzero (L2 2.282). All four keep ratios are legal in the real FastV hook. None was removed.

## Stage I

On the reloaded pristine60 images, r=0.25 strict is 52/60 for B. The paired mean of Delta_0.25 (B minus noise) is 17.940. Its bootstrap 95% interval is [15.763, 20.242]. Every strict success has Delta_0.25 > 0. The strict-group median is 17.730 and the non-strict median is 10.888. Gates A, B, and C pass.

The same quantities at the other frozen budgets are not monotone. Strict counts are 11/60 at r=0.50, 48/60 at r=0.35, 52/60 at r=0.25, and 31/60 at r=0.15. Zero samples have Delta increasing at every step as the keep ratio falls. A monotone budget gate is not supported.

Gate D fails. The check is that Full-text equality and the teacher-forced hard-margin cone agree. They disagree for samples 5615, 1023, 22317, 51140, 56084, and 60744. Seven of the eight disagreeing image conditions have identical Full text while at least one clean-sequence hard margin is not strictly positive. One does not: reloaded noise for 60744 keeps every hard margin positive, but the greedy Full text is `SIMON FRASER UNIVERSITY` while the clean Full text is `Simon Fraser University`.

## Not run

Calibration, the 12-sample three-arm comparison, a new pristine60 draw, formal B+ attack, repeat10, and the 2x2 mechanism audit were not run.

## Claims

These artifacts support only a stopped diagnosis: the compression-conditional margin gap is present at every frozen budget, and the budget curve is not monotone. They do not support a B+ method, a new generalization rate, or a claim about all compressors or models.

## Integrity

raw_immutability_check.json is PASS for all 2858 recorded historical files. FROZEN_V7_PROTOCOL.json was written once and then verified.
