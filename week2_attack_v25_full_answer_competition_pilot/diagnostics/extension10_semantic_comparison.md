# Fixed-selector semantic endpoint comparison

This comparison uses the final PNG after 10 accepted updates at `r=0.25`.
The two attack objectives use exactly the same step size, pixel budget, Full
guard, replay protocol, and accepted-update budget. Random controls are the
three existing matched `L_inf=8` PNGs from the frozen v20 control set.

| sample | Native-FDP | complete-target | random controls |
|---|---|---|---|
| 47450 | `CONTENT_FAILURE` | `CONTENT_FAILURE` | 0/3 content failures |
| 7006 | `ANSWER_PRESERVED` | `ANSWER_PRESERVED` | 0/3 content failures |

Both attack endpoints on 47450 output the same complete wrong answer,
`THE NUTRITION FOUNDATION, INC.`, while preserving Full text/token exactness,
Full replay, FastV replay, and `uint8 L_inf=8`. On 7006, neither attack
produced a content failure. Every random control preserved the clean FastV
answer and passed the Full checks.

Therefore the fixed-selector comparison shows a real, non-random attack on one
sample, but no incremental advantage for `complete-target` over Native-FDP on
these two samples. The complete-target loss should be retained as a targeted
attack baseline and not claimed as a superior method from this result.

