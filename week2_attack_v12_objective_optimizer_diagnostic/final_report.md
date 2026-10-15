# Objective/optimizer four-cell smoke report

Status: `BLOCKED_SMOKE_GATE`

This run executed the six-group deterministic replacement on two development
samples (`1025`, `60700`) for three updates per group. The groups were:

- H0/H1: dynamic/fixed FastV reference with Adam;
- P0/P1: dynamic/fixed reference with weighted PGD;
- O0/O1: dynamic/fixed reference with orthogonal/selective PGD.

All jobs used fresh Python processes, GPUs 0/1/4/6, deterministic CUDA/PyTorch
settings, and `PYTHONNOUSERSITE=1` so the environment-local
`huggingface_hub==0.36.2` was used instead of the incompatible user-site
`2.1.1`. The first P0/O0 launch was rejected before model loading by that user
site package mismatch and was rerun without changing the environment.

Smoke endpoint summary:

| group | 1025 Full exact all steps | 60700 Full exact all steps |
|---|---:|---:|
| H0 dynamic + Adam | pass | pass |
| H1 fixed + Adam | pass | pass |
| P0 dynamic + PGD | pass | pass |
| P1 fixed + PGD | pass | pass |
| O0 dynamic + OPGD | pass | pass |
| O1 fixed + OPGD | pass | **fail at step 3** |

The failing trajectory is
`runs/smoke/O1/60700/trajectory.json`. At step 3 it has
`full_text_exact=0`, although the output is valid, the PNG replay is still
consistent, and the uint8 L-infinity bound passes. This is a genuine failure
of the Full-preservation gate for this optimizer/reference combination, not an
instrumentation error. No six-sample run and no unseen run was started.

The successful cells also produced no joint FastV success within three steps;
that is not evidence of attack failure because three steps is only a smoke
budget. The smoke gate is solely a correctness/stability gate.

Artifacts are under `runs/smoke/` and the launch logs are `smoke_*log`.
Because the gate failed, the next authorized action is to diagnose or repair
the OPGD Full-preservation rule (for example, reject/rollback an update whose
post-update Full text changes), then rerun the two-sample smoke. Do not select a
winner, run the six development samples, or touch the frozen 20-sample unseen
cohort until every cell passes.
