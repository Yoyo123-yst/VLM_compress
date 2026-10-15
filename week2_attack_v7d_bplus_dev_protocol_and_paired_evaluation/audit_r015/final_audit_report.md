# r=0.15 strict audit

Decision: `PASS_R015_RECONCILED_STRICT_DEFINITION_DRIFT`.

The 31/60 and 45/60 counts label the same 60 B images and the same Full and FastV strings. They are not interchangeable. Later B-Native, native-FDP, and B+ results use only the canonical strict in `canonical_strict_spec.json`.

## strict_31_of_60

Source: `week2_attack_v7_budget_conditioned/stage1_diagnose.py`.

```
name != clean
AND pred_full == same_run_clean_full_text
AND anls_full >= 0.5
AND anls_fastv(r=0.15) < 0.5
AND output_valid
AND fastv text nonempty
AND 0 < uint8_linf <= 8/255
AND same_run clean anls_full >= 0.5
AND same_run clean anls_fastv(r=0.15) >= 0.5
```

Full is exact text against the clean generate from that same stage1 run, and also ANLS >= 0.5. FastV is ANLS < 0.5 plus a nonempty string. The extra gate is the clean FastV ANLS measured at r=0.15 in that same run.

## strict_45_of_60

Source: `week2_attack_v7c_native60_readonly_remeasurement/merge.py` `remeasured_strict`.

```
status in {SELECTED, empty}
AND anls_full >= 0.5
AND anls_fastv < 0.5
AND full_text == manifest clean_full_pred
AND both texts output_valid
AND published replay_ok
AND 0 < published uint8_linf <= 8/255
AND manifest clean_full_anls >= 0.5
AND manifest clean_fastv_anls >= 0.5
```

Full is exact text against the manifest clean string, and also ANLS >= 0.5. FastV is ANLS < 0.5. Replay, valid, and L∞ are required. The clean FastV ANLS is the historical manifest value, not a clean generate at r=0.15.

## The 14 flips

Ids: 1023, 2288, 7937, 8122, 18736, 25716, 41828, 44779, 49432, 50803, 51298, 56403, 58827, 62650.

Every one of these has the same PNG SHA-256, the same selector step, the same Full string, and the same FastV string. Old strict is 0 and v7c strict is 1. The stage1 clean FastV ANLS at r=0.15 is 0.0 on all 14, so the stage1 clause `clean anls_fastv(r=0.15) >= 0.5` fails and v7c's historical clean FastV ANLS clause does not. No other clause is required to explain the flip. 31 + 14 = 45.

## Canonical strict used from here

```
valid_output
AND nonzero uint8 perturbation
AND uint8 L∞ <= 8/255 + 1e-6
AND Full generate text == clean Full generate text
AND FastV ANLS < 0.5
AND disk reload replay matches
```

Full exact is decoded-text equality. FastV error is ANLS < 0.5. Native margin, cone_holds, and a hand-added EOS do not enter this predicate.
