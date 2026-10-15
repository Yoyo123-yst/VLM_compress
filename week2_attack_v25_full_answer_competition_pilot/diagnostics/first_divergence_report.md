# First-divergence diagnostic

## Scope

This is a read-only diagnostic over the already accepted PNGs from `iterative_v2`.
It evaluates clean, accepted steps 1, 3, and 5 for samples 47450 and 7006 at
`r=0.25`. The PNGs are reloaded in independent processes, the natural FastV
keep-set is recomputed, and teacher-forced logits are scored under that same
keep-set. No hooks, wrappers, or training-state modifications are used.

All 6 trajectories had previously passed the Full text exact, Full token exact,
Full replay, FastV replay, and uint8 `L_inf <= 8` guards for every accepted
update. The present diagnostic does not replace those endpoint guards.

## What was measured

At every answer position we recorded the clean-answer token, the frozen complete
wrong-answer token, and the strongest competitor excluding those two. The key
quantities are:

- `wrong - correct`: whether the wrong target has caught the correct token;
- `wrong - competitor`: whether the wrong target has beaten all other local
  competitors;
- suffix means after the first target-position difference.

## Results

### 47450 (`Dr. or Mrs. William J. Darby` -> `THE NUTRITION FOUNDATION, INC.`)

The first target position is position 0. `wrong - correct` changes as follows:

| method | clean | step 1 | step 3 | step 5 |
|---|---:|---:|---:|---:|
| Native-FDP | -3.375 | -3.000 | -1.812 | -0.562 |
| complete-target | -3.375 | -2.250 | -0.875 | -0.188 |
| alias-group | -3.375 | -2.875 | -0.813 | -0.438 |

The wrong target beats the strongest non-target competitor by step 3/5, but
still does not beat the correct first token. The suffix `wrong - correct` mean
stays strongly negative (about -16 to -18) and does not improve in a way that
would support a naturally generated complete answer. FastV remains exactly
`Dr. or Mrs. William J. Darby` at all inspected steps.

### 7006 (`RJ Reynolds Tobacco Company` -> `Public Relations Dept.`)

The first target position is position 0:

| method | clean | step 1 | step 3 | step 5 |
|---|---:|---:|---:|---:|
| Native-FDP | -2.438 | -2.000 | -1.812 | -1.688 |
| complete-target | -2.438 | -1.812 | -1.500 | -1.188 |
| alias-group | -2.438 | -1.812 | -1.375 | -1.125 |

For complete-target, the wrong target is only marginally above the strongest
non-target competitor at step 5 (`+0.062`), while it remains `1.188` log-prob
units below the correct first token. The suffix mean remains around `-14` and
FastV remains exactly `RJ Reynolds Tobacco Company`.

## Interpretation

The complete-answer objective is moving the *first divergence margin* in the
intended direction, especially on 47450. It is not yet making the wrong answer
the winner: the wrong target can beat incidental competitors while losing to
the correct first token. The later target positions are not the bottleneck
being solved; they are scored under a prefix that natural decoding never
enters.

This rules out the claim that the five-step pilot already induces a complete
wrong answer. It does support the next pre-registered branch of the protocol:
keep the same target and guard, and extend the fixed-budget trajectory to
10--25 accepted updates. Do not add a new selector, loss term, or optimizer in
that run.

## Next gate

Run only `complete_target` and `native_fdp` on these two samples, with the same
PNG replay and Full exact guards, first to 10 accepted updates. Continue to 25
only if the first-position `wrong - correct` gap keeps shrinking and no replay
or determinism check fails. A natural `CONTENT_FAILURE` must be observed in
the independently reloaded PNG; surrogate loss or teacher-forced target
scores alone are insufficient.

