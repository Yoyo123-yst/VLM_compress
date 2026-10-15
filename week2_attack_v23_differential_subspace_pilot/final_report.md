# Differential-subspace one-step pilot

Status: `NO_GO_DIFFERENTIAL_SUBSPACE_PILOT`

This was a four-sample, one-step diagnostic at FastV ratio 0.25. It compared
RAW FastV content gradients, the current Native-FDP joint gradient, a local
compressed-versus-Full differential direction, and a matched random direction.
Each direction was evaluated at uint8 L-infinity radii 1, 2, 4, and 8 after
PNG save/reload. No multistep training or unseen evaluation was started.

## Protocol checks

- The existing deterministic `processor.prepare` path was reused.
- Every endpoint was evaluated after PNG reload.
- Full text exact, Full token exact, Full replay, natural FastV text, keep-set,
  processor hash, and per-token margins were recorded.
- The first implementation exposed a sign mismatch between the continuous
  generalized direction and the sign-PGD pixel update. The direction was
  reoriented using the actual `sign(direction)` update and the pilot was rerun.
  Only the corrected run is included in the result files.

## Results

The differential direction was not consistently better than the existing
attack direction.

| sample | observation |
|---|---|
| 47450 | DIFF lowered the core margin, but not below RAW; DIFF failed Full exact at radius 4 and produced no content failure. |
| 60700 | DIFF lowered the margin relative to RANDOM, but remained worse than RAW; DIFF failed Full exact at radius 4 and did not produce a confirmed content failure. |
| 7006 | DIFF was worse than RAW and RANDOM at every tested radius; the natural answer did not change. |
| 39204 | DIFF was slightly below RAW at some radii, but the natural answer remained `FOOD TECHNOLOGY` and no content failure occurred. |

The matched random direction did not create a content failure. The observed
margin changes therefore do not establish a new attack mechanism.

## Decision

The pilot does not provide the required discriminator witness:

1. DIFF did not beat RAW/NATIVE at a fixed Full-safe endpoint.
2. It did not rescue a Native-FDP failure sample with a confirmed content
   replacement.
3. It was not uniformly safer for Full exact replay.

The differential-subspace route is stopped at the diagnostic stage. No
multistep refresh, checkpoint selector, development expansion, or unseen run
is authorized by this result.

The valid scientific conclusion is narrower: a local sensitivity ratio can
change the surrogate margin, but the present construction does not turn that
change into a reliable Full-preserving semantic attack advantage.
