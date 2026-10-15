# Complete-front small development validation

## Frozen question and cohort

Test whether `complete-front` adds value beyond `complete-target` when both receive the same clean-generated, semantically reviewed complete wrong answer. Native-FDP is retained only as the ordinary content-attack reference. This is development evidence, not unseen evaluation.

The six IDs and order are frozen in `sample_selection_freeze.json` before reading their questions, labels, images, or model outputs. Keep all six in the denominator; never replace a sample after clean screening or target generation.

## Frozen implementation

- Keep the v27 code path, model, processor, page ordering, auxiliary-page role, ratio 0.25, PNG format, `epsilon=8/255`, `alpha=1/255`, Full-FDP coefficient 1, target-generation code and first-valid-target rule unchanged.
- Candidate generation is clean-only, capped at 24 raw branches and 6 complete continuations per sample. Do not use dataset answer text, ANLS, attack outcomes, or held-out information to rank candidates. Human/answer-type review happens after candidates are generated and before attack runs.
- For complete-target and complete-front, freeze identical correct-expression tokens and the exact same wrong-target tokens.
- Native-FDP does not use a wrong target; score content failure separately from target hit.
- Each method gets a fixed backward-pass budget of 40: Native-FDP 20 attempts x 2 backward passes; complete-target 13 attempts x 3; complete-front 10 attempts x 4. The final unused pass (if any) is not reassigned. Every rejected attempt consumes its full planned backward cost.
- The runner must execute the fixed number of attempts, not stop early when it reaches an accepted-update count. Keep every rejection and all counters.
- Endpoint selector: last accepted PNG at the end of the fixed attempt schedule. If none was accepted, mark `NO_VALID_SEALED_CANDIDATE`; do not promote an earlier intermediate success as the final endpoint.
- No method-specific retries, extra steps, target substitutions, or oracle checkpoint selection.

## Gates before attack

1. Restore each frozen sample and verify question, answer page, page order, page hashes, evidence/auxiliary indices, expected FastV K, and processor zero-delta equality.
2. Clean Full and FastV must be valid and semantically correct under the frozen audit rule. Otherwise keep the sample as `CLEAN_GATE_FAILED`, do not attack it, and retain it in the six-sample denominator.
3. Generate no more than the frozen candidate cap; review answer type and whether the target is actually distinct from all accepted aliases. If none passes, record `NO_VALID_PREREGISTERED_COMPLETE_WRONG_TARGET`; do not replace the sample.
4. Before launch, compare the code/config hashes against this protocol and run one sample/method short smoke plus deterministic replay. Any mismatch blocks the full run.

## Scoring and evaluator

- Count `CONTENT_FAILURE`, `TARGET_HIT`, clean answer type, and `INVALID_OUTPUT` separately.
- `CONTENT_FAILURE` requires question-specific semantic disagreement with the dataset answer; spelling/case/format-only changes remain preserved or ambiguous.
- `TARGET_HIT` counts only if the frozen target passed semantic answer-type review and the natural output from the selected PNG matches its normalized exact target.
- Report first success cost and final endpoint separately. Cost includes every attempt, including rejected updates: backward calls, score forwards, generation calls, and wall time.
- The attack process may use in-process replay only as a guard. Final PNGs must be evaluated by a separate Python process with a fresh model load. The evaluator records Full text/token exact, replay, FastV text, semantic status, ANLS, K, keep-set hash, PNG hash, and uint8 L-infinity.
- Do not open or evaluate any of the reserved 20 samples. Sample 452 is explicitly contaminated and cannot be called untouched.

## Decision rule

- Keep complete-front only if, on valid clean-start samples with valid targets, it either turns a complete-target final miss into a final `TARGET_HIT`/`CONTENT_FAILURE`, or repeatedly reaches the same qualifying endpoint at lower actual cost under the fixed budget.
- If complete-target is equal or better, keep the simpler complete-target method.
- If the only gain is exact target hit without higher semantic content-failure rate, limit the claim to targeted answer control.
- If clean admission or target coverage is low, that is a reported limitation; do not redraw the cohort.
