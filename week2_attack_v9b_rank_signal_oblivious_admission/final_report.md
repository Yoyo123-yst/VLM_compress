# v9b rank-signal-oblivious gray-box admission

Decision: `BLOCKED_BASELINE_OR_SURROGATE`

Phase 2 status: `BLOCKED_NO_ADMISSIBLE_RANK_OBLIVIOUS_SURROGATE`

This is a five-question development admission. It is not an unseen result and not a generalization result.

## Baseline

Phase 0 recovered the Full baseline. The prompt is the historical `build_prompt` from `week1_mechanism/run_eval.py`, including the `<|im_start|>user` / `<|im_end|>` / `<|im_start|>assistant` wrapper. Sample 1025 generates `Dr. C. Baur MD` with tokens `8847, 13, 356, 13, 425, 4110, 13979, 151645`, and the three pages still contain 11097 image tokens. All five admission ids match their frozen Full text. A new process reloaded the saved pages and reproduced the same text and token ids. Status: `BASELINE_RECOVERED`.

The five ids are the pre-registered prefix `1025, 2008, 7006, 39204, 47450`.

## Why training did not start

Both target-independent stochastic bottleneck surrogates entered the real multi-page generate path, changed the token count, reloaded from disk, and kept FastV access at 0. The frozen clean gate still requires surrogate ANLS of at least 0.5 at ratios 0.25, 0.35, and 0.50 on every admission id. That gate was not lowered.

Failures:

- SpatialStridedDrop: 1025 at 0.25 (`Dr. MD`, ANLS 0) and 0.35 (`Dr. M.D.`, ANLS 0); 2008 at 0.25 (`January 1968`, ANLS 0).
- LocalSpatialMerge: 1025 at 0.25 (`Dr. BMD`, ANLS 0); 2008 at 0.25 (`February 23, 1968`, ANLS 0).

Both families are `SURROGATE_NOT_ADMITTED`. Training, PNG sealing, and target FastV evaluation are `NOT_RUN`. There is no gray-box success rate from this stage.

No memory overflow occurred. The clean gate used GPUs 0, 1, 4, and 6, one sample per card.

## What this does not change

Historical white-box B stays at its frozen 52/60 result. It was not rerun.

White-box Native-FDP remains the earlier development upper bound, strict 10/12 on the twelve-question dev set. It was not rerun and is not a gray-box result.

This stage is only the rank-signal-oblivious gray-box admission. It did not produce a transfer rate. A zero transfer rate was not measured, because FastV was not called.

No unseen holdout was started.
