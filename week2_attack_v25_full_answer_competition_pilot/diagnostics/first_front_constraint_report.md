# First-divergence constraint comparison

## Setup

The only change from `complete_target` is a hinge at the first position where
the frozen wrong answer differs from the clean answer:

`relu(tau + max_logit(excluding_wrong) - wrong_logit)`

with frozen `tau=0.25` and weight `beta=1.0`. The target answer, Full-FDP,
step size, pixel budget, PNG quantization, selector, and replay guards are
unchanged. The comparison uses 10 accepted updates at `r=0.25`.

## Results

| sample | Native-FDP first success | complete-target first success | complete-front first success |
|---|---:|---:|---:|
| 47450 | step 7 | step 6 | step 4 |
| 7006 | no success by step 10 | no success by step 10 | step 8 |

The first success means the independently reloaded PNG naturally generated the
frozen complete wrong answer. All accepted updates in the new method passed
Full text exact, Full token exact, Full replay, FastV replay, and `uint8
L_inf <= 8`.

For 7006, the new method changes the endpoint from the clean answer
`RJ Reynolds Tobacco Company` to `Public Relations Dept.`. A fresh process
replayed the step-10 PNG and obtained the same result. Its step-10 first-token
margin was `wrong - correct = +0.125` and `wrong - strongest-competitor =
+0.562`.

For 47450, the new method reaches the same wrong answer as both baselines, but
earlier. Matched random controls at the same `L_inf=8` budget preserve the clean
answer on both samples.

The extra hinge adds one differentiable front-position scoring/backward pass per
update. In these runs, the complete-front wall time was about 333 s on 47450
and 259 s on 7006 for 10 updates; the corresponding complete-target runs took
about 303 s and 232 s. Wall time is hardware-run dependent, but the extra
backward pass is a real cost and must be included in any later comparison.

## Cost and interpretation

The added hinge requires an additional differentiable scoring pass, so wall
time per accepted update is higher. The gain on 7006 is therefore meaningful
only if it remains after reporting equal forward/backward/generation cost. This
two-sample result is a discriminator witness for the mechanism, not yet a
general method result: it shows that forcing the first natural decision can
rescue one failure sample, while the original complete-answer objective cannot.

The next controlled experiment should repeat the three methods on the
pre-registered clean-start development samples 60700 and 39204 with the same
10-step budget and fixed selector. Expand only if the first-divergence method
continues to rescue baseline failures without sacrificing Full exactness or
requiring an unfair compute budget.

