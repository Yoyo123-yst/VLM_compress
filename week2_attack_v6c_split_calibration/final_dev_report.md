# v6c split calibration

Decision: `PASS_V6C_DEV40_AWAITING_PRISTINE60_CONFIRMATION`.

The selected version is Variant B. Variant C was trained on the 12-sample development set and was not selected. v6 and v6b stay blocked. Their 17 hashed protocol and result files are unchanged. pristine60 was not started.

## Coefficients

`lambda_soft` is the immutable v6b quotient `2.049824826423478`. It was not recomputed.

`beta` comes from one auxiliary-only probe on sample 1025, `delta = +1/255`. The reloaded uint8 L∞ is `1/255`. At that state, compression gradient L2 is `8.68726634979248` and selection gradient L2 is `0.4353623688220978`, so `beta = 19.954104837532157`. The clean-image selection gradient is 0, which this protocol treats as expected. There was no second probe.

## Development set

A remains strict 7/12, pollution 1, median keep Jaccard `0.8355726620046946`.

B on the 12 selected images: strict 11/12, Full exact 12/12, replay 12/12, nonzero 12/12, uint8 L∞ 12/12, invalid 0, pollution 1. The only non-strict id is 50779. Among the 11 strict successes the exclusive labels are CONTENT_SUFFICIENT 8, BOTH_INDEPENDENTLY_SUFFICIENT 2, INTERACTION_REQUIRED 1, SELECTION_SUFFICIENT 0.

C: strict 10/12, pollution 2. It misses the common pollution cap and its strict count is below B, so the frozen rule does not select it. Median keep Jaccard is `0.831`, also below A. C therefore does not support a selection-stable claim.

## Full 40 confirmation

Variant B was rerun on the existing 40-sample manifest, four shards, 200 steps, the frozen lambda, and no beta. Each shard finished 10/10 with status `ok`. Logs contain no traceback, OOM, or non-finite gradient. The wrapper exit lines are no longer in the terminal store; completion is taken from those logs and the 40 `ok` rows.

The prospective selector kept a nonzero, L∞-valid, Full-exact checkpoint for 40/40. Disk replay matched the stored Full and FastV text for 40/40.

| Check | Result |
| --- | --- |
| Strict | 38/40 |
| Full exact | 40/40 |
| Replay | 40/40 |
| Nonzero | 40/40 |
| uint8 L∞ | 40/40 |
| Invalid | 0 |
| Matched v5b noise | 3/40 |
| Attack-only / noise-only | 35 / 0 |
| Exact McNemar two-sided p | 5.82e-11 |

The two non-strict ids are 47450 (`DR OR Ms WILLIAM J DARABBY`, FastV ANLS 0.885) and 52248 (`$525,000`, FastV ANLS 0.667). Both keep the clean Full text.

The allowed description is a source-bounded compression-only attack, with content-sufficient evidence on the development successes. It is not a selection-stable survivor poisoning result.

pristine60 is waiting for an explicit confirmation. It has not been launched.
