# First-divergence extension: 10 accepted updates

## Protocol result

The pre-registered extension used the same two frozen complete-answer targets,
the same sign-PGD step (`1/255`), the same pixel budget (`uint8 L_inf <= 8`),
and the same Full/Fast replay guards. It ran `native_fdp` and `complete_target`
on 47450 and 7006 in fresh processes. All four trajectories reached 10/10
accepted updates. Every accepted update passed Full text exact, Full token exact,
Full replay, FastV replay, and the pixel bound. The original v2 result hashes
were unchanged.

## Endpoint observations

### 47450

At step 10, both methods produced the independently reloaded natural FastV
answer `THE NUTRITION FOUNDATION, INC.` while Full remained exact. The first
position margin (`wrong - correct`) was:

| method | clean | step 5 | step 10 |
|---|---:|---:|---:|
| Native-FDP | -3.375 | -0.562 | +0.062 |
| complete-target | -3.375 | -0.188 | +1.937 |

Thus the complete-answer target moves the first target token more strongly, but
it does not create a unique success: Native-FDP reaches the same wrong answer by
step 10. The suffix target margins remain strongly negative, which is expected
because the later target tokens are scored under teacher-forced prefixes that
natural decoding only visits after the first token has switched.

Matched random-noise controls from the existing frozen v20 control set used the
same uint8 `L_inf=8` budget. All three 47450 controls kept the clean answer and
were `ANSWER_PRESERVED`; therefore the 47450 endpoint is not explained by the
matched random control.

### 7006

Neither method switched naturally by step 10. For `complete_target`, the first
position reached `wrong - correct = -0.125` and `wrong - competitor = +0.562`,
so the wrong target beat incidental competitors but still lost to the correct
first token. The natural answer remained `RJ Reynolds Tobacco Company`.
The three existing matched random controls also preserved that answer.

## Decision

The first-divergence hypothesis is supported as a diagnostic explanation:
complete-answer optimization can make the wrong first token win the local
competition, and once it wins, natural decoding follows the complete wrong
answer. However, this is not evidence for a new method yet, because Native-FDP
also succeeds on 47450 and the method does not rescue 7006.

Do not expand to 25 steps or unseen samples on the basis of this result. The
next valid comparison is a fixed-selector, equal-budget semantic evaluation on
the two samples (including the matched random controls), followed by a decision
whether complete-target has any advantage over Native-FDP. If there is no
advantage on additional clean-start samples, keep it as a useful targeted
baseline and stop claiming a novel attack mechanism.

