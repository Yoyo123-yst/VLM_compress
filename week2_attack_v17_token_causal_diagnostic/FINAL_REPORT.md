# Token/keep-set causal diagnostic

## Scope

This is a read-only diagnostic on development sample `60700`, ratio `r=0.25`.
It replays two existing PNGs in one fresh model process:

- `step_011.png`: FastV output `PPSB` (the successful error state);
- corrected attack-only `step_001.png`: FastV output `PPBS` (the recovery state).

No pixels were updated, no checkpoint was selected, and no unseen result was
produced.

## Replay gate

Both images passed real Full checks:

- Full text: `PPBS`;
- Full token exact: pass;
- Full replay: pass.

The endpoint FastV outputs were stable within this process:

| image | FastV output | FastV ANLS | keep-set hash |
|---|---|---:|---|
| start error | `PPSB` | 0.0 | `94ff1513...70508` |
| recovery step 1 | `PPBS` | 1.0 | `0b13da85...e868b297` |

## Four-way teacher-forcing diagnostic

The reference error sequence is `PPSB`, with token pieces `PP`, `SB`, EOS.
For each image, margins were evaluated under the generated/reference sequence
and the start/current keep-set. These are teacher-forced logits, not endpoint
claims.

On the start error image, all four combinations were identical:

```text
PP: hard=1.000, soft=0.306
SB: hard=0.125, soft=0.085
EOS: hard=3.375, soft=2.436
```

On the recovery image:

```text
                         PP             SB             EOS
reference + current   1.500/0.912   -1.750/-1.767   3.938/3.186
reference + start     1.938/1.048   -0.500/-0.537   4.125/3.187
generated + current   1.500/0.912    1.750/1.654   2.125/1.850
generated + start     1.938/1.048    0.500/0.441   2.625/2.310
```

The first number is hard margin and the second is soft margin. A positive
margin means the observed token beats its strongest rival.

## Interpretation

1. The `SB` token is stable enough in the error PNG (`+0.125`) but becomes
   strongly noncompetitive on the recovery PNG (`-1.75`, or `-0.5` even when
   paired with the old keep-set).
2. Changing the keep-set alone changes the margin quantitatively but does not
   restore the `SB` token. Therefore the current evidence does **not** support
   a Top-K-boundary or keep-set-stability explanation.
3. The recovery image has a different strong competitor at the second
   position (`7347`, piece `S`, margin `+1.75`). This is a genuine autoregressive
   token-state switch, not merely a harmless capitalization change.
4. The earlier “freeze after first success” result is consistent with this:
   continuing the Full-FDP updates erodes the already-achieved `SB` margin.
   Freezing is a valid state-retention protocol, but by itself it is not a new
   attack mechanism.

## Decision

`NO_GO_KEEP_SET_STABILITY_ROUTE`.

Do not expand the Top-K stability loss, cross-budget state machine, or another
generic selector construction from this sample.

## Next falsifiable experiment

The only justified follow-up is a small targeted persistence test, not a full
run:

1. Use the fixed error sequence `PPSB` and target the `SB` position with a
   standard candidate-vs-original competition loss.
2. Keep the existing Native-FDP Full exact guard and uint8 L∞ budget.
3. Once an unforced PNG first produces `PPSB`, save that PNG immediately and
   stop changing pixels; continue only endpoint replay to step 25.
4. Compare against Native-FDP and the same targeted loss without freezing on
   `60700` and one recovery-heavy sample (`39204` or `7006`).

This experiment can establish whether the actionable bottleneck is error-state
retention. If freezing is the only improvement, report it as an evaluation /
state-selection protocol rather than a novel loss. If the targeted persistence
loss still cannot hold the token under fixed Full guard, stop this route and do
not add more objectives.

Artifacts:

- `causal_report.json`
- `run_token_causal.py`
