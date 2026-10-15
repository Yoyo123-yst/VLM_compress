# v19 diagnostic protocol repair

## Status

`FIXED_SMOKE_PASS` for `60700_error`, one update, both natural and frozen
selection modes. A repeated fresh-process run produced identical endpoint,
gradient, projection and page hashes.

## Repairs

1. The attack reference is now the clean-input FastV sequence. The Full clean
   sequence remains separate and is used only by Native-FDP/Full exact guard.
2. Clean, start-state, gradient and endpoint evaluation all reconstruct pixels
   through `pixels_from_page`; no path uses the stale `inputs["pixel_values"]`
   for the auxiliary page.
3. The start-state frozen keep-set is taken from the same page-derived pixels
   used by the gradient path.
4. Each update records the keep-set/tokens actually used by the gradient, plus
   pre-update, projected-floating-point and PNG-reloaded surrogate snapshots.

## Smoke evidence

Artifacts:

- `results_fixed_smoke.json`
- `results_fixed_smoke_repeat.json`

For both modes, step 0 has the same clean FastV reference, start FastV tokens,
keep-set hash, gradient hash, projected delta hash and endpoint output across
the two fresh processes. Full exact/replay and the uint8 perturbation guard
passed for the accepted candidate. The natural and frozen snapshots now agree
at the initial state; their post-update losses are intentionally different
because frozen mode holds the initial keep-set while natural mode reselects.

## Scope boundary

This repairs the diagnostic measurement protocol. It is not evidence that a
new attack mechanism works, and it does not authorize the previous full
three-state or six-sample rerun. The next permitted run is the same finite
diagnostic on the remaining states, with the repaired fields checked before
any method conclusion.

## Follow-up states

The repaired run was completed for `60700_recovery` and `7006_near` with two
selection modes, four step sizes, and four updates (`80` endpoint records).
All records passed Full exact/replay and the uint8 perturbation bound.

- `60700_recovery`: the clean answer `PPBS` remained unchanged at the two
  smallest step sizes. At `1/255` and `2/255`, final endpoints became
  `PPBS 14, 1967` or `Ppjs 14, 1967`, both ANLS `0` under the registered
  metric. This reproduces a short-step attack transition, but not yet a claim
  about semantic failure because the output retains the core token `PPBS`.
- `7006_near`: the start answer was `Tobacco Company` (ANLS about `0.577`).
  Every setting either stayed there or returned to `RJ Reynolds Tobacco
  Company` (ANLS about `0.963`); no strict endpoint was reached.

Thus the old qualitative split persists: the recovery state is locally
steerable under larger finite steps, while the near-answer state remains
semantically stable. This is diagnostic evidence, not evidence for a new
method. The next analysis must separate metric-only suffix changes from true
content errors before expanding the experiment.

## Endpoint reclassification

`reclassify_endpoints.py` was run on the repaired JSON without loading a
model. The policy uses the registered normalized answer core and compares each
trajectory with its own step-0 state, so a pre-existing near-answer mismatch
is not counted as an attack-induced failure.

- Overall endpoint labels: 36 `ANSWER_PRESERVED_EXACT`, 1
  `ANSWER_PRESERVED_AUGMENTED`, and 43 `CONTENT_FAILURE` under the strict
  ground-truth-core rule.
- `7006_near` is already a baseline content mismatch (`Tobacco Company` does
  not contain the registered `RJReynolds Tobacco Company` core), so it has
  `0` induced content failures.
- `60700_recovery` has `3` induced content-failure records, all at the largest
  step (`2/255`) and all producing `Ppjs` rather than `PPBS`. The
  `PPBS 14, 1967` endpoints are correctly classified as answer-preserving
  augmented outputs rather than failures.

These three records are a useful diagnostic witness, but they are not yet a
general attack result: the induced change is a near-spelling substitution and
the sample is a single recovery state. No new training run is authorized by
this reclassification alone.

## Core-margin GPU diagnostic

The second diagnostic was run in parallel on GPU 0 (`60700_recovery`) and GPU
6 (`7006_near`). Each of the 80 records now contains clean-reference loss,
PNG-reloaded loss, core-token hard margins, natural text, and natural/used
keep-set hashes.

- `60700_recovery`: the clean FastV reference span was located exactly as
  tokens `[4406, 7347]` (`PPBS`). For all step sizes, the attack loss and the
  core margin generally decreased together. At the largest step, the margin
  reached roughly `1.09` from a start of `1.41` in the natural path and
  produced `Ppjs`; however, smaller or intermediate updates still generated
  `PPBS`. The surrogate is therefore locally aligned with weakening the
  reference tokens, but not sufficient to guarantee a semantic switch.
- `7006_near`: the manifest spelling and tokenizer segmentation differ, so
  the clean reference span was conservatively recorded as all five non-EOS
  reference tokens `[49, 41, 45072, 65110, 8188]`. Across natural and frozen
  paths, PNG-reloaded attack loss stayed flat or increased (about `2.395` at
  start, up to roughly `2.86`), while the core margin increased from `2.075`
  to as high as roughly `2.90`. The generated answer alternated between
  `Tobacco Company` and `RJ Reynolds Tobacco Company`; no content failure was
  induced.

The diagnostic therefore does not support a single global explanation. On
60700 the surrogate can weaken core-token margins but often remains inside the
same semantic basin. On 7006 the actual PNG update does not even weaken the
clean-reference core margin. The next method change should target answer-core
competition and quantized/selected-path behavior, but must first be tested as
a baseline against these two distinct failure modes.
