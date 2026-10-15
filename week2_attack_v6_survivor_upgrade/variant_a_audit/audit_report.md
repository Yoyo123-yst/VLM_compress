# Variant A audit

Decision: `PASS_VARIANT_A_AUDIT`.

The selector ranks saved v5b checkpoints by compressed margin, then earlier step. A candidate is eligible only when step > 0, the reloaded uint8 image differs from clean, uint8 L∞ is at most 8/255, the checkpoint output is valid, and the Full text matches the clean Full text. ANLS, FastV text, ground truth, and replay are not ranking inputs. Replay runs after the pick. No sample needed a second pick.

## Dev-12 gate

| Check | Result |
| --- | --- |
| Full exact | 12/12 |
| Disk replay | 12/12 |
| Nonzero uint8 | 12/12 |
| uint8 L∞ | 12/12 |
| Invalid output | 0 |
| Strict | 7/12 |
| v5b raw hash | PASS, 212 files, 0 changed |

Strict ids: 1025, 2008, 47567, 51133, 60700, 6194, 64506.

Selected steps that are not 200: 7006 and 64506 at step 150, 60700 at step 100. Every selected dev image has pixel L∞ exactly 8/255.

## Mechanism on the selected PNGs

Exclusive labels on all 12: CONTENT_SUFFICIENT 5, BOTH_INDEPENDENTLY_SUFFICIENT 2, INTERACTION_REQUIRED 1, SELECTION_SUFFICIENT 0, UNEXPLAINED_OR_INVALID 4.

Among the 7 strict successes: CONTENT 4, BOTH 2, INTERACTION 1. Inclusive content sufficiency is 6/7. Pollution among those 7 is 1, sample 60700 (`PPBS` → `Ppbs 14, 15c, d`).

The four unexplained samples are not strict successes. Their attacked FastV text still meets ANLS 0.5. Every cell had a non-empty decode, and the regenerated Full text matched both the selector row and the clean Full text.

## All 40

40/40 had a feasible nonzero checkpoint. Disk replay matched the stored Full and FastV text on 40/40. Post-replay strict success is 35/40. That 40-sample count is an audit export, not the stage-I gate and not a new holdout.
