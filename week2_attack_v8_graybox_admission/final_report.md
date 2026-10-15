# Gray-box admission

Decision: `BLOCKED_NO_ADMISSIBLE_SURROGATE`.

This stage did not optimize pixels, did not run PGD, and did not call target FastV. There is no new strict rate. Historical B 52/60, Native60, and the v7/v7d files were not rewritten. Parent hashes are in `parent_hash_manifest_before.json` and `parent_hash_manifest_after.json`.

## Identity

`MATERIALLY_DIFFERENT`.

The executed B_NATIVE Full term is a soft margin pulled toward a lowered decision floor. The executed Native-FDP Full term is a one-sided hard-margin penalty against the clean hard margin, teacher-forced on the real Full generate tokens, with no added EOS. Both still add the target FastV compressed loss, so Native-FDP remains a white-box oracle. The 10/12 development strict is not a gray-box result. B_PLUS stays the ablation where native Delta rose and real strict did not.

## Surrogates

Twelve local candidates are listed in `surrogate_candidate_audit.csv`. Zero are admitted.

A candidate is not admitted when it is the target, when its selector is only FastV-family attention Top-K, or when it lacks a LLaVA-OneVision multi-page generate path, a clean gate, or a replay. Token-batch rankers and the CAGE VisionZip stack are visible and are still not admitted. Visibility is not admission.

## Isolation and the next step

The denylist, import counters, and separate train/evaluate processes in `isolation_design.md` are specific enough to audit later. They were not executed. The 3–5 question transfer draft stays a draft. It must not run until a real surrogate passes the gates this stage did not find.
