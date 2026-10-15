# Corrected direction and selection-stability audit

## Scope

This is a diagnostic construction on development sample 60700 at `r=0.25`.
It is not a method result and was not transferred to unseen data. The starting
PNG is the previously sealed legal error state with Full=`PPBS` and FastV=`PPSB`.

All candidates use real FastV generation, real keep-set extraction, PNG replay,
and a Full text/token/replay exact guard. Gradients, PNG construction, and
guards use the same `base_page + delta` state.

## Results

| variant | step 0 | step 1 | steps 2--3 | Full exact |
|---|---|---|---|---|
| corrected error-margin + Full-FDP | `PPSB` | `PPBS` | `PPBS` | 4/4 |
| corrected error-margin only | `PPSB` | `PPBS` | `PPSB` | 4/4 |
| corrected error-margin + Top-K boundary stability | `PPSB` | `PPBS` | `PPBS` | 4/4 |

The attack-only run oscillated back to `PPSB`; neither objective kept the
natural error state. The selection-stability term did not provide a witness:
the first update already returned to the Full-like FastV answer.

## Decision

`NO_GO_SELECTION_STABILITY_CONSTRUCTION`.

The earlier v15 error-state result was invalid as a mechanism test because its
margin sign and gradient/evaluation pixel state were inconsistent. After those
corrections, the error state still fails to persist, and adding a differentiable
Top-K boundary gap does not fix it. This is evidence that simply stabilizing the
observed keep-set is not sufficient; no claim about a new attack mechanism is
supported.

Do not expand this route to 7006, the six-sample development set, or unseen
evaluation. The valid positive result remains the sealed 60700 PNG itself,
which is a checkpoint/endpoint result rather than a new optimizer.
