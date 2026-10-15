# v9c multiscale rank-oblivious admission

Decision: `GRAYBOX_TRANSFER_EVALUATED`

This is a five-question development transfer. It is not an unseen result.

## Baseline and structure

Full baseline stayed recovered. All five ids matched the frozen Full text and token ids, including 1025 as `Dr. C. Baur MD` with tokens `8847, 13, 356, 13, 425, 4110, 13979, 151645`. Each page has 3699 visual tokens. A second process reloaded the PNGs and reproduced the same text and tokens.

CPU structure tests passed before the language model was loaded. The frozen pools are 14×13, 21×26, and 33×33 for ratios 0.25, 0.35, and 0.50. Both families are labeled `not_an_official_compressor`.

## Clean admission

BasePreservingAdaptivePool passed 15/15 conditions and was admitted.

BasePreservingJitteredPool passed 42/45. It failed 1025 at ratio 0.25 offset 2, and at ratio 0.50 offsets 0 and 1. The family was not admitted. No id, ratio, or offset was removed after the result.

## Training and seal

Training ran for 200 Adam steps on the admitted family only. Lambdas were frozen from the 1025 `+1/255` probe before training: `lambda_fdp = 0.2486486700704209`, `lambda_reg = 2062.940181825187`. The training process did not import FastV.

Four PNGs were sealed: 1025 step 150, 2008 step 100, 7006 step 50, 47450 step 25. Sample 39204 had no checkpoint that kept the Full token sequence, so its status is `NO_VALID_SEALED_CANDIDATE` and it was not sent to FastV.

## Isolated FastV

FastV ran only after the seal, in a separate process. On the five-question development set:

| ratio | strict |
| --- | --- |
| 0.15 | 1/5 |
| 0.25 | 1/5 |
| 0.35 | 0/5 |
| 0.50 | 0/5 |

The single strict case is 1025, at 0.15 (`TAKEDA`, ANLS 0) and 0.25 (`Dr. Bacharach`, ANLS 0). At 0.35 and 0.50 that sample stays close to the clean answer. The other three sealed samples keep FastV ANLS at or above 0.5 at every evaluated ratio. Full exact, valid output, disk replay, and the L∞ bound hold for the four sealed samples, which is 4/5 of the development set. The remaining sample was not evaluated.

Surrogate optimization therefore transferred to FastV on one development question at the primary ratio 0.25. It did not produce a FastV failure on the other sealed questions, or on 1025 at 0.35 and 0.50.

Historical white-box B and Native-FDP were not rerun.
