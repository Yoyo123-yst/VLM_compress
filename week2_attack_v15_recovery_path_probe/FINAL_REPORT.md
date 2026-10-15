# Recovery-path construction report

## Decision

`NO_GO_RECOVERY_PATH_CONSTRUCTION` on the registered 7006, `r=0.25` probe.
The experiment is diagnostic/construction only; it is not a development-set
method result and was not transferred to unseen data.

## Scope

- sample: `7006`
- ratio: `0.25`
- steps: 25 accepted-update attempts
- B: content non-overlap loss on the clean FastV prefix
- C: the same loss with a bounded replay buffer of naturally observed
  recovery prefixes (maximum 3)
- Full exact guard after every proposed update
- auxiliary-page-only perturbation, uint8 L-infinity budget `8/255`

The original trajectory files had an evaluation bug: `anls_score` was called
with a string although its API requires a list of answers. The runner and the
diagnostic were corrected, and both B and C were rerun from fresh processes.
Only the corrected rerun is used below.

## Corrected endpoint table

| method | step 0 | first low-overlap output | step 25 | Full exact |
|---|---|---|---|---|
| B | `RJ Reynolds Tobacco Company` | step 10: `Tobacco Company` (ANLS vs ground truth 0.577) | `Tobacco Company` (0.577) | 25/25 guards passed |
| C | `RJ Reynolds Tobacco Company` | step 5: `Tobacco Company` (0.577) | `Tobacco Company` (0.577) | 25/25 guards passed |

The clean answer is `RJReynolds Tobacco Company`; strict failure requires
ANLS `< 0.5` against that ground-truth answer. Neither B nor C reaches that
endpoint. Both outputs remain semantically close (`Tobacco Company`), and B
recovers to the full company name at steps 15/20; C recovers at steps 10/15/20
before returning to the same near-answer at step 25.

## Interpretation

The run confirms that the content-level objective can move generation into a
near-answer basin while preserving Full, but it does not establish a strict
FastV failure. The replay-buffer variant C does not provide a discriminator
over B: it is not more successful, does not produce a lower ANLS endpoint, and
does not prevent recovery. Therefore the proposed recovery-path coverage
mechanism has no construction witness and must not be expanded to more samples
or unseen evaluation.

The 39204 `r=0.35` diagnostic likewise did not show a strong recovery-path
failure pattern, so it cannot rescue this route.

## Error-state strengthening follow-up

Using the existing legal 60700 `r=0.25` state (`Full=PPBS`, `FastV=PPSB`),
the complete observed error sequence `PPSB` was optimized for 10 additional
updates with the same Full guard. The starting PNG was a strict failure, but
the first update returned FastV to `PPBS`; steps 2--10 stayed at `PPBS` or
case-only `Ppbs`. Full exact/replay passed at every step. Thus sequence
strengthening did not preserve an already successful natural error state.

This closes the proposed sequence-strengthening rescue for the current
objective. The only reliable operation observed here is sealing the first
successful PNG before further updates; that is a checkpoint protocol, not an
attack mechanism. The follow-up trajectory is in
`construction/60700/ERROR_SEQ_STRENGTHEN/trajectory.json`.

Artifacts:

- `construction/7006/B/trajectory.json`
- `construction/7006/C/trajectory.json`
- `diagnostics/7006_r25.json`
- `diagnostics/39204_r35.json`
- `diagnostics/60700_r25.json`

