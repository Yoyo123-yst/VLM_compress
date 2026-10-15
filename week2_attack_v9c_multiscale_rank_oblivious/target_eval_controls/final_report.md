# v9c target-side control

Decision: `GRAYBOX_CONTROL_NOT_SUPPORTED`.

This is a 5-sample development post-hoc control. It does not support unseen generalization, budget-independent transfer, or compressor-oblivious generalization. Confirming the 1025 clean-to-attack drop would not change that scope, and the drop is not specific to the sealed attack PNG.

## What was held fixed

Training code, the surrogate, the loss, hyperparameters, the training seed, candidate selection, and the four sealed PNGs were not changed. No new attack candidate was trained. `target_eval/fastv_transfer.json` was not overwritten. FastV ran only in this control process. The access counter incremented on each real `run_fastv` call: 104 calls. Full generate calls: 26. Token IDs are not returned by `run_plain` or `run_fastv`, so they are null. The exactness claim is pre-seal token exact plus target-side text exact.

Prompt, processor, page order, and decode match the v9c target evaluator: template A from `week1_mechanism/run_eval.py::build_prompt`, greedy generate, `max_new_tokens=16`, `use_cache=True`. Precheck passed: clean page hashes match the manifest, and the four sealed PNG hashes match `seal_manifest.json`.

## Controls

Clean FastV was measured on all five original pages at r in {0.15, 0.25, 0.35, 0.50}. The historical r=0.25 clean screen matches this rerun on text, ANLS, K, and keep-set hash for every ID, including 1025 (`Dr. C. Baur`, ANLS 0.6667, K=2774).

The four sealed attack PNGs were re-evaluated at the same ratios. Every FastV text matches `target_eval/fastv_transfer.json`. Status is not `REPLAY_DIVERGED`. Full text stayed equal to the frozen clean Full text, and disk replay matched.

Matched noise uses an independent uint8 perturbation on the same auxiliary page, drawn uniformly from integers in [-8, 8], clipped to [0, 255], and saved through the same PNG path. The seed is `sha256("v9c-control-noise-2026101121|" + sample_id)` and was fixed before FastV ran. It does not read FastV output and does not copy the attack residual. Actual L∞ is 8/255 on all four noise PNGs. Noise PNG hashes differ from the sealed attack hashes. Other pages stay equal to the clean pages. 39204 has no sealed candidate, so both the attack and noise branches are `NO_VALID_SEALED_CANDIDATE`. That sample stays in the overall denominator.

Clean strict means valid output, Full text exact, replay match, and FastV ANLS < 0.5. It does not require a perturbation. Attack strict and random strict add a nonzero uint8 change and L∞ ≤ 8/255.

## Per-ratio counts

Overall denominator is 5. Conditional sealed counts use denominator 4 and do not replace the overall counts. Coverage is 4/5. 39204 is unmeasured on attack and noise; those cells are missing, not zeros. Induced failure requires clean FastV ANLS ≥ 0.5 and attack FastV ANLS < 0.5.

| ratio | clean strict | attack strict | random strict | attack−clean ANLS (measured) | attack−random ANLS (measured) | induced failures |
| --- | --- | --- | --- | --- | --- | --- |
| 0.15 | 1/5 | 1/5 | 1/5 | 1025: 0; 2008: 0; 7006: 0; 47450: −0.08 | 1025: 0; 2008: 0; 7006: 0; 47450: −0.08 | none; 39204 unmeasured |
| 0.25 | 0/5 | 1/5 | 1/5 | 1025: −0.667; 2008: +0.20; 7006: 0; 47450: 0 | 1025: 0; 2008: +0.20; 7006: 0; 47450: 0 | 1025 only; 39204 unmeasured |
| 0.35 | 0/5 | 0/5 | 0/5 | 1025: 0; 2008: +0.20; 7006: 0; 47450: 0 | 1025: 0; 2008: +0.20; 7006: 0; 47450: 0 | none; 39204 unmeasured |
| 0.50 | 0/5 | 0/5 | 0/5 | 1025: +0.167; 2008: 0; 7006: 0; 47450: 0 | 1025: +0.167; 2008: 0; 7006: 0; 47450: 0 | none; 39204 unmeasured |

Conditional sealed attack strict is 1/4, 1/4, 0/4, 0/4 at these four ratios.

The r=0.15 clean strict count is 1025. Its clean FastV text is already `TAKEDA` (ANLS 0). The attack text is also `TAKEDA`. That cell is not an induced failure.

## Sample 1025

At r=0.25, clean FastV is `Dr. C. Baur` (ANLS 0.6667) and the sealed attack is `Dr. Bacharach` (ANLS 0). Full text remains `Dr. C. Baur MD`, and replay matches. Relative to clean, this is an induced failure.

The matched-noise page at the same L∞ produces the same FastV text, `Dr. Bacharach`, and the same ANLS 0. Attack ANLS is not lower than noise ANLS. The sealed PNG therefore does not beat the noise control. At r=0.15, clean already fails, so that attack result is not an induced failure either. At r=0.50 the attack ANLS (0.9167) is higher than both clean and noise (0.75).

## Why the decision is not supported

`GRAYBOX_CONTROL_CONFIRMED` would require the r=0.25 1025 clean-to-attack induced failure together with a result that is not explained by matched noise. The induced failure and the Full/replay checks are present. The noise control reproduces the same failure, so the result does not support attributing that failure to the optimized attack PNG.

No required cell is missing except the pre-registered coverage failure on 39204. Replay did not diverge.
