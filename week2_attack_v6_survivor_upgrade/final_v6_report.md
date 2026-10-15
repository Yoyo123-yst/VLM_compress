# v6 survivor upgrade

Decision: `BLOCKED_BC_IMPLEMENTATION_PREREQUISITE`.

B and C were not trained. pristine60 was not started. v5b raw files remain unchanged: 212 checked, 0 changed.

## Stage 0

Parent protocol `FROZEN_DEV_PROTOCOL.json` SHA-256 is `2e9a05f5d1be778de6321f783a99c790d85f75eae37e17e1370805fb66de0c87`. It was not overwritten. The addendum records the same 12 development ids, epsilon 8/255, 200 Adam steps at learning rate 2/255, and the real checkpoint steps 0, 10, 25, 50, 75, 100, 150, 200.

## Stage I

`PASS_VARIANT_A_AUDIT`.

The selector now uses reloaded uint8 pixels. Nonzero means pixel L∞ > 0. Ranking uses only step, nonzero, valid output, exact Full text, uint8 L∞, and compressed margin. Replay is after the pick. No sample was repicked.

On the 12 development ids the selected images are Full exact 12/12, replay 12/12, nonzero 12/12, uint8 L∞ 12/12, invalid output 0, strict 7/12. The strict ids are 1025, 2008, 47567, 51133, 60700, 6194, and 64506.

Exclusive mechanism labels on those 7 are CONTENT_SUFFICIENT 4, BOTH_INDEPENDENTLY_SUFFICIENT 2, INTERACTION_REQUIRED 1. Pollution among them is 1: sample 60700, `PPBS` to `Ppbs 14, 15c, d`. The other four development samples are not strict successes; their FastV text still meets ANLS 0.5.

Across the existing 40, the corrected selector finds a feasible image for 40/40, disk replay matches 40/40, and strict success is 35/40. That count is an audit of saved v5b checkpoints, not a new holdout.

## Stage II

Health on sample 1025, one model on GPU 0, passed the implementation checks that can be checked before a coefficient exists:

- clean Full and FastV text match the stored clean outputs;
- K is 2774 and the generate keep set matches the real FastV hook;
- the checkpointed last-query scores match `score_prompt` with max absolute difference 0;
- the answer sequence ends with EOS and has length 8;
- evidence-page pixel gradient L2 is 0;
- compression gradient L2 is about 9.47;
- losses are finite.

The Full token margin is differentiable. Its gradient L2 at the clean image is about 9.18. The frozen protection term is not that margin. It is `relu(kappa_t - m_t)` with `kappa_t = min(max(m_clean_t, 0), 1)`. All eight clean margins are above 1, so every kappa is 1 and every hinge input is negative. The hinge gradient is exactly 0. The one allowed +1/255 shift still gives hinge gradient L2 0. Selection mass does get a gradient there, L2 about 0.435. There is no legal `lambda_seq`, and the protocol does not allow a second probe or a hand-set coefficient.

## GPUs

At the stop, GPUs 0, 1, 4, and 6 were free. GPUs 2, 3, 5, and 7 were occupied by other jobs and were not used. Each of our jobs used one free GPU. The earlier out-of-memory error was one uncheckpointed score backward filling a single 40 GB card. After checkpointing, health and calibration fit on GPU 0. Extra cards do not create a nonzero hinge gradient.

## pristine60

Do not start it from this run. B and C have no frozen coefficient, so they cannot be the version taken forward. Variant A's 35/40 is a re-reading of the existing development images, not a new unseen set. A later pristine60 would be a new protocol decision about the already measured source-bounded attack, not a continuation of this blocked B/C scale rule.
