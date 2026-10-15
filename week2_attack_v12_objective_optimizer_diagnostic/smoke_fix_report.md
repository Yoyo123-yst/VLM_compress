# Corrected smoke report

Status: `SMOKE_GATE_PASS`

The OPGD projection order was corrected from `project_out(full, attack)` to
`project_out(attack, full)`. Every update now also receives a real Full
`generate()` endpoint check; an update that changes the Full text is restored
from the pre-update delta.

The two-sample smoke used fresh processes on GPUs 0/1/4/6, samples `1025` and
`60700`, six groups, and three updates per group. All 12 trajectories passed
Full exact, Full replay, valid output, and the pixel bound at every saved
checkpoint. The aggregate gate is therefore `PASS`.

The corrected OPGD case `O1/60700` did require one rollback at update step 2:
the attempted update failed the real Full exact check, was rejected, and the
next checkpoint remained Full-exact. This is a successful safety-gate result,
not evidence that the unconstrained OPGD proposal is safe.

No cell produced a joint FastV attack success within this three-step smoke
budget. That is expected for a correctness smoke and is not an attack-result
claim. The six-sample development run and the frozen 20-sample unseen run have
not been started.

Artifacts: `runs/smoke_fix/` and `smoke_fix_*log`.
