# Multi-seed noise diagnostic

Decision: `GRAYBOX_SIGNAL_NOT_ESTABLISHED_ABOVE_RANDOM`.

This is a 5-sample development diagnostic. It is not unseen generalization, not budget-independent transfer, not cross-compressor transfer, and not evidence that the gray-box attack is generally effective. One sealed attack failure is not a statistical test against the random ensemble.

Consistency flags: none.
FastV calls in the evaluation process: 660.
Full gate pass among 160 sealed noise PNGs: 160.
Overall denominator remains 5. Sealed-attack coverage is 4/5. Sample 39204 stays `NO_VALID_SEALED_CANDIDATE` and is not removed.

## Random induced-failure rate

Induced means the clean page is not an endpoint failure, the noise passed the Full gate, and the noise FastV output is valid with ANLS < 0.5. Sample 1025 at r=0.15 is a clean endpoint failure, so its noise failures are not induced.

### r=0.15
Pooled passing controls: 0/160 = 0.0000, Wilson 95% CI [0.0000, 0.0234].
- 1025: clean_failed=True, gate 32/32, induced 0/32 = 0.0000, CI [0.0000, 0.1072].
- 2008: clean_failed=False, gate 32/32, induced 0/32 = 0.0000, CI [0.0000, 0.1072].
- 7006: clean_failed=False, gate 32/32, induced 0/32 = 0.0000, CI [0.0000, 0.1072].
- 47450: clean_failed=False, gate 32/32, induced 0/32 = 0.0000, CI [0.0000, 0.1072].
- 39204: clean_failed=False, gate 32/32, induced 0/32 = 0.0000, CI [0.0000, 0.1072].

### r=0.25
Pooled passing controls: 28/160 = 0.1750, Wilson 95% CI [0.1239, 0.2413].
- 1025: clean_failed=False, gate 32/32, induced 28/32 = 0.8750, CI [0.7193, 0.9503].
- 2008: clean_failed=False, gate 32/32, induced 0/32 = 0.0000, CI [0.0000, 0.1072].
- 7006: clean_failed=False, gate 32/32, induced 0/32 = 0.0000, CI [0.0000, 0.1072].
- 47450: clean_failed=False, gate 32/32, induced 0/32 = 0.0000, CI [0.0000, 0.1072].
- 39204: clean_failed=False, gate 32/32, induced 0/32 = 0.0000, CI [0.0000, 0.1072].

### r=0.35
Pooled passing controls: 0/160 = 0.0000, Wilson 95% CI [0.0000, 0.0234].
- 1025: clean_failed=False, gate 32/32, induced 0/32 = 0.0000, CI [0.0000, 0.1072].
- 2008: clean_failed=False, gate 32/32, induced 0/32 = 0.0000, CI [0.0000, 0.1072].
- 7006: clean_failed=False, gate 32/32, induced 0/32 = 0.0000, CI [0.0000, 0.1072].
- 47450: clean_failed=False, gate 32/32, induced 0/32 = 0.0000, CI [0.0000, 0.1072].
- 39204: clean_failed=False, gate 32/32, induced 0/32 = 0.0000, CI [0.0000, 0.1072].

### r=0.50
Pooled passing controls: 0/160 = 0.0000, Wilson 95% CI [0.0000, 0.0234].
- 1025: clean_failed=False, gate 32/32, induced 0/32 = 0.0000, CI [0.0000, 0.1072].
- 2008: clean_failed=False, gate 32/32, induced 0/32 = 0.0000, CI [0.0000, 0.1072].
- 7006: clean_failed=False, gate 32/32, induced 0/32 = 0.0000, CI [0.0000, 0.1072].
- 47450: clean_failed=False, gate 32/32, induced 0/32 = 0.0000, CI [0.0000, 0.1072].
- 39204: clean_failed=False, gate 32/32, induced 0/32 = 0.0000, CI [0.0000, 0.1072].

## Sample 1025, r=0.25

Clean: `Dr. C. Baur` ANLS 0.6666666666666667.
Sealed attack, not rerun: `Dr. Bacharach` ANLS 0.0; induced=True.
Random induced failures: 28/32 = 0.8750, Wilson 95% CI [0.7193, 0.9503].

| seed_index | seed | text | ANLS | endpoint | induced | gate |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 2027126122 | Dr. Bacharach | 0.0 | True | True | PASS |
| 1 | 2027126123 | Dr. Bacharach | 0.0 | True | True | PASS |
| 2 | 2027126124 | Dr. Bacharach | 0.0 | True | True | PASS |
| 3 | 2027126125 | Dr. Bacharach | 0.0 | True | True | PASS |
| 4 | 2027126126 | Dr. Bacharach | 0.0 | True | True | PASS |
| 5 | 2027126127 | Dr. Bacharach | 0.0 | True | True | PASS |
| 6 | 2027126128 | Dr. Bacharach | 0.0 | True | True | PASS |
| 7 | 2027126129 | Dr. Bacharach | 0.0 | True | True | PASS |
| 8 | 2027126130 | Dr. C. Baur | 0.6666666666666667 | False | False | PASS |
| 9 | 2027126131 | Dr. Bacharach | 0.0 | True | True | PASS |
| 10 | 2027126132 | Dr. C. B. M. D. | 0.75 | False | False | PASS |
| 11 | 2027126133 | Dr. C. Baur | 0.6666666666666667 | False | False | PASS |
| 12 | 2027126134 | Dr. Bacharach | 0.0 | True | True | PASS |
| 13 | 2027126135 | Dr. Bacharach | 0.0 | True | True | PASS |
| 14 | 2027126136 | Dr. Bacharach | 0.0 | True | True | PASS |
| 15 | 2027126137 | Dr. Bacharach | 0.0 | True | True | PASS |
| 16 | 2027126138 | Dr. Bacharach | 0.0 | True | True | PASS |
| 17 | 2027126139 | Dr. Bacharach | 0.0 | True | True | PASS |
| 18 | 2027126140 | Dr. Bacharach | 0.0 | True | True | PASS |
| 19 | 2027126141 | Dr. Bacharach | 0.0 | True | True | PASS |
| 20 | 2027126142 | Dr. Bacharach | 0.0 | True | True | PASS |
| 21 | 2027126143 | Dr. Bacharach | 0.0 | True | True | PASS |
| 22 | 2027126144 | Dr. Bacharach | 0.0 | True | True | PASS |
| 23 | 2027126145 | Dr. Bacharach | 0.0 | True | True | PASS |
| 24 | 2027126146 | Dr. Bacharach | 0.0 | True | True | PASS |
| 25 | 2027126147 | Dr. Bacharach | 0.0 | True | True | PASS |
| 26 | 2027126148 | Dr. Bacharach | 0.0 | True | True | PASS |
| 27 | 2027126149 | Dr. Bacharach | 0.0 | True | True | PASS |
| 28 | 2027126150 | Dr. Bacharach | 0.0 | True | True | PASS |
| 29 | 2027126151 | Dr. Bacharach | 0.0 | True | True | PASS |
| 30 | 2027126152 | Dr. C. Baur | 0.6666666666666667 | False | False | PASS |
| 31 | 2027126153 | Dr. Bacharach | 0.0 | True | True | PASS |

Attack-induced cells and their random Wilson intervals are listed in `summary.json`. The attack PNG was not rerun and was not modified.
