# Attack-compensation probe

Decision: `NO_GO_ATTACK_COMPENSATION_STRUCTURE`.

This is a 6-sample development constructability probe. It is not a method result, not an unseen evaluation, and not evidence that the construction beats Native-FDP.

## Baseline

Clean Full text, clean FastV text/K/keep hash at r=0.25 and r=0.35, and the historical Native-FDP PNG replay matched the v7d records. See `baseline_replay.json`.

## Witnesses

Denominator 6. Strong 0/6. Strong or partial 0/6. Witness ids: none.
Sample 1025 is marked noise-sensitive at r=0.25 from the earlier 28/32 matched-noise diagnostic and cannot decide the result alone.

| sample | Native-FDP strict | status | witness | accepted | rejected |
| --- | --- | --- | --- | --- | --- |
| 1025 | 1 | OK | none | 100 | 0 |
| 2008 | 1 | NO_VALID_COMPENSATED_CANDIDATE | none | 100 | 0 |
| 47450 | 1 | OK | none | 100 | 0 |
| 60700 | 1 | OK | none | 100 | 0 |
| 7006 | 0 | OK | none | 100 | 0 |
| 39204 | 0 | NO_VALID_COMPENSATED_CANDIDATE | none | 100 | 0 |

## Ratio conditions

### 1025
- r=0.25: full_break=False, fastv_fail=True, recovers_full=True, combined_fastv_fail=True, comp_benign=False, control_not_equivalent=True.
- r=0.35: full_break=False, fastv_fail=True, recovers_full=True, combined_fastv_fail=True, comp_benign=True, control_not_equivalent=True.

### 2008
- r=0.25: not measured.
- r=0.35: not measured.

### 47450
- r=0.25: full_break=False, fastv_fail=True, recovers_full=True, combined_fastv_fail=True, comp_benign=True, control_not_equivalent=False.
- r=0.35: full_break=False, fastv_fail=True, recovers_full=True, combined_fastv_fail=True, comp_benign=True, control_not_equivalent=False.

### 60700
- r=0.25: full_break=True, fastv_fail=True, recovers_full=True, combined_fastv_fail=False, comp_benign=True, control_not_equivalent=True.
- r=0.35: full_break=True, fastv_fail=True, recovers_full=True, combined_fastv_fail=False, comp_benign=True, control_not_equivalent=True.

### 7006
- r=0.25: full_break=False, fastv_fail=False, recovers_full=True, combined_fastv_fail=False, comp_benign=True, control_not_equivalent=True.
- r=0.35: full_break=False, fastv_fail=False, recovers_full=True, combined_fastv_fail=False, comp_benign=True, control_not_equivalent=True.

### 39204
- r=0.25: not measured.
- r=0.35: not measured.

## Endpoint detail

Attack components were found for all 6 samples. Two samples, 2008 and 39204, have no Full-exact compensated checkpoint, so the six-way witness is not measured there. Their attack-only endpoints were still scored.

- 1025: attack keeps Full `Dr. C. Baur MD` and breaks FastV (`CHADWICK, C.` / `CHICHAYLES`, ANLS 0). Compensation step 10 keeps Full and keeps the FastV failure. Compensation alone is not benign at r=0.25: it outputs `Dr. Bacharach` (ANLS 0). Sign-flip and roll both change Full to `Dr. C. Baur`.
- 2008: attack changes Full to `EOR March 23, 1953` and breaks FastV (`Dec. 1938` / `January, 1923`). All 100 compensation updates were accepted by the continuous constraint, but no checkpoint restored Full text. Status `NO_VALID_COMPENSATED_CANDIDATE`. Controls were not run.
- 47450: the selected compensation checkpoint is step 0, so the compensation delta is zero. Attack keeps Full and breaks FastV (`TREASURER` / `Walmart`). Sign-flip and roll match that same attack image, so the controls are equivalent.
- 60700: attack changes Full from `PPBS` to `PPD3` and breaks FastV. Compensation step 25 restores Full to `PPBS` and also restores FastV to `Ppbs` (ANLS 1) at both train ratios. Compensation alone stays benign. Sign-flip changes Full to `District Health Objectives`.
- 7006: attack does not change Full or FastV (`RJ Reynolds Tobacco Company`, ANLS 0.963). The selected compensation checkpoint is step 0, so there is no learned compensation.
- 39204: attack changes Full from `FOOD TECHNOLOGY` to `Food Technology` and does not make FastV fail (ANLS 1.0 at r=0.25, 0.75 at r=0.35). No Full-exact compensated checkpoint. Controls were not run.

Strong witnesses 0/6. Partial witnesses 0/6. The only sample that breaks both Full and FastV and then restores Full is 60700, and that compensation also repairs FastV. 1025 does not break Full, and its compensation-only image is itself a FastV failure at r=0.25.

## What was not run

Native-FDP was not retrained. Fixed dual loss, alternating optimization, generic gradient projection, and dynamic attack-compensation role assignment are NOT_RUN.

## Cost

- 1025: counters={'forward': 9158, 'backward': 300, 'generate': 1092, 'fastv': 1054}, elapsed_sec=2745.0087274389807, peak_mib=26415.0732421875
- 2008: counters={'forward': 8902, 'backward': 300, 'generate': 1070, 'fastv': 1040}, elapsed_sec=2111.209100524895, peak_mib=23100.09619140625
- 47450: counters={'forward': 11908, 'backward': 300, 'generate': 1110, 'fastv': 1072}, elapsed_sec=3770.3707216219045, peak_mib=30420.60986328125
- 60700: counters={'forward': 5804, 'backward': 300, 'generate': 1102, 'fastv': 1064}, elapsed_sec=2643.1612924518995, peak_mib=26928.447265625
- 7006: counters={'forward': 8009, 'backward': 300, 'generate': 1110, 'fastv': 1072}, elapsed_sec=3512.973972273059, peak_mib=30763.271484375
- 39204: counters={'forward': 5913, 'backward': 300, 'generate': 1070, 'fastv': 1040}, elapsed_sec=3280.666820114013, peak_mib=30766.55859375
