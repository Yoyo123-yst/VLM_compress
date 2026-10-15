# Smoke gate after sequential-ratio backward memory fix

Status: `PASS_SMOKE_MEMFIX2`

The smoke cohort contains 3 samples (`1025`, `60700`, `47450`) and all six
groups (`H0`, `H1`, `P0`, `P1`, `O0`, `O1`), for 18 trajectories total. Each
trajectory ran for 3 update steps in a fresh process with the deterministic
runtime settings and the real post-update Full exact/replay guard.

## Results

- Completed: 18/18
- Full text exact at every checkpoint: 18/18
- Full replay: passed for every recorded checkpoint
- PNG uint8 constraint: passed for every trajectory
- Unexpected OOM: 0
- Maximum recorded peak memory: 29,084.5 MiB (`47450`)
- Guard rollbacks: 1 (`60700/O1`), accepted by the guard; no trajectory lost
  Full exactness
- Joint FastV attack success in this 3-step smoke: 0/18 (this gate tests
  stability and resource feasibility, not attack efficacy)

The implementation change is limited to `attack_grads`: the two training
ratios backpropagate sequentially into the same leaf gradient, avoiding
simultaneous retention of both FastV graphs. The mathematical gradient target
is the same sum, but floating-point accumulation order changed, so this smoke
gate is required before comparing against historical trajectories.

## Decision

The memory blocker is cleared for this configuration. The next permitted run
is the six-sample development four-cell experiment under a new tag. No unseen
sample has been run, and no historical-result replacement is implied.
