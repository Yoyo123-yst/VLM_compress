# Future transfer admission protocol

Status: draft only. Do not execute it while the decision is `BLOCKED_NO_ADMISSIBLE_SURROGATE`.

## Samples

If a later stage is unblocked, the training ids are the first five ids of the already published v7d development list, in that list's order:

`1025, 2008, 7006, 39204, 47450`

They are not reordered by a new hash, not replaced, and not taken from pristine60 or any unseen pool. A 3-question prefix of the same list is the only smaller legal subset. No other development id may be added after seeing a transfer result.

## Order

1. Surrogate training finishes inside `graybox_train/`.
2. The trainer writes PNGs, image hashes, config hash, and the access log, then exits.
3. A new process confirms `target_fastv_access_count == 0` and an empty module list.
4. `target_fastv_evaluate/` loads the sealed PNGs and calls target FastV for the first time.
5. Strict labels are computed only in that evaluator.

There is no query to target FastV before step 4. A failed seal or a nonzero access count stops the evaluator.

## Gates

Every sample must have nonzero uint8 perturbation, uint8 L∞ at most 8/255, Full text exact, and disk-reload replay. Invalid output must be zero.

At least one sample must be a real target-FastV strict success. Strict still means FastV ANLS below 0.5 together with the checks above. Margin does not count.

If strict is 0/5, the decision is `NO_GO_SURROGATE_TRANSFER_ZERO`. That outcome does not open a 40-question or 60-question run.

An unseen cohort may be created only after this admission passes, and only as a new id list that does not reuse historical evaluation ids. This draft does not create that list.
