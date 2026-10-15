# v11c prospective holdout transfer

Status: `TRANSFER_EVALUATED`

This is a holdout relative to the current v11c six-sample selection. The IDs
come from the previously frozen pristine60/calibration pool, so this is not an
unseen-dataset generalization result and must not be reported as one.

## Protocol

- 20 frozen IDs, selected before this run and never used to tune v11c.
- 200 deterministic Native-FDP steps per sample.
- All 20 selected checkpoints passed Full exact, valid output, nonzero pixel
  perturbation, and uint8 L-infinity `8/255`.
- Independent disk-reloaded clean/attack/matched-noise FastV evaluation.
- Ratios: `0.15`, `0.25`, `0.35`, `0.50`.
- Every branch was evaluated twice from disk.

## Results

| metric | count |
|---|---:|
| samples trained | 20/20 |
| samples with valid selected checkpoint | 20/20 |
| FastV cells | 80 |
| all replay checks passed | 80/80 |
| induced cells (`clean >= 0.5`, `attack < 0.5`) | 42/80 |
| attack strict cells | 45/80 |
| matched-noise strict cells | 3/80 |
| induced cells also beating noise | 42/42 |
| samples with at least one induced ratio | 19/20 |

Induced counts by ratio were `7/20` at r=0.15, `17/20` at r=0.25,
`17/20` at r=0.35, and `1/20` at r=0.50. Sample 62321 had no induced
ratio; the other 19 samples had at least one.

## Interpretation boundary

The current run shows that the deterministic v11c Native-FDP construction can
produce Full-preserving FastV failures on this prospective pool, and the
induced cells all beat the matched-noise ANLS control. It does not establish
unseen generalization, compressor-oblivious gray-box transfer, or recovery of
the historical v7d trajectory. The pool was inherited from the earlier
pristine60/calibration development universe and therefore remains a
development/selection-adjacent result.
