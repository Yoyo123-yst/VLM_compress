# v7d development comparison

Decision: `SELECT_NATIVE_FDP`.

This is a 12-question development selection. It is not a generalization result. No unseen cohort, no 40/60-question extension, and no B+ generalization run was started.

## What was held fixed

Historical Variant B at r=0.25 remains 52/60. This stage did not recompute or rewrite that endpoint. Parent trees are unchanged: 3360 files before and after, `raw_immutability_check.json` pass.

The r=0.15 counts 31/60 and 45/60 come from two different strict predicates on the same 60 images and the same generated strings. Stage 1 decision: `PASS_R015_RECONCILED_STRICT_DEFINITION_DRIFT`. Those two counts are not interchangeable. Every new method below uses only the canonical predicate: valid output, nonzero uint8 perturbation, uint8 L∞ at most 8/255, Full generate text identical to the clean Full text, FastV ANLS below 0.5, and a disk-reload replay match. On this development set the FastV clause is required at both training ratios, r=0.25 and r=0.35. Native margin is used for the loss and for Delta, not for the strict label.

Stage 2 decision: `PASS_DIFFERENTIABLE_FASTV_FULL_SEQUENCE_ADMISSION` on the frozen 10 conditions.

Coefficients were frozen before these development outcomes, from calibration sample 51070 at a single +1/255 probe: `lambda_full` = 2.6371882367154997, `lambda_gap` = 0.913170951688196, xi = 1, eta = 0.001. Training used only r = 0.25 and r = 0.35. r = 0.50 and r = 0.15 were scored after selection and were not used to train or to pick the method.

## Development result

All three methods ran 12 questions for 200 steps. All 36 jobs exited 0. Disk replay, Full exact, nonzero perturbation, and the L∞ bound are 12/12 for every method. Invalid outputs are 0. Output pollution is 0 for every method.

Canonical strict, requiring FastV ANLS < 0.5 at both r=0.25 and r=0.35:

| Method | Strict | Mean train-band native Delta |
| --- | ---: | ---: |
| B_NATIVE | 6/12 | 5.811 |
| NATIVE_FDP | 10/12 | 5.715 |
| B_PLUS | 9/12 | 9.161 |

Against B_NATIVE, NATIVE_FDP gains 2008, 50779, 51133, and 60700, and loses none. B_PLUS gains 2008, 51133, and 60700, and loses none. B_PLUS therefore meets its extra admission checks against B_NATIVE: strict is higher, gains exceed losses, and train-band Delta is higher. It does not outrank NATIVE_FDP. The frozen order places strict count ahead of Delta. 10/12 is ahead of 9/12. B_PLUS's larger Delta is a mechanism difference, not an additional strict success.

Per-ratio strict, not used as a substitute for the conjunction above:

- B_NATIVE: r=0.50 1/12, r=0.35 6/12, r=0.25 9/12, r=0.15 7/12.
- NATIVE_FDP: r=0.50 0/12, r=0.35 10/12, r=0.25 11/12, r=0.15 8/12.
- B_PLUS: r=0.50 1/12, r=0.35 9/12, r=0.25 10/12, r=0.15 5/12.

No method is `NOT_RUN`.

## Stop

Monitoring and training are stopped after this report. The selected method is NATIVE_FDP on this development set only.
