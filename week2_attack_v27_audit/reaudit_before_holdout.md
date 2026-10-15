# Pre-holdout result and protocol correction

This file is an audit addendum. It does not rewrite or delete the original v27 records.

## Counting rules frozen for the next comparison

- `CONTENT_FAILURE`: natural FastV output fails a question-specific, predeclared semantic check against the dataset answer. A spelling/case/format change alone is not enough.
- `TARGET_HIT`: the natural FastV output matches the frozen complete target under exact normalized text. This is reported separately from content failure; target answer type and semantic validity must pass review before it counts.
- `ANSWER_TYPE_VALID`: the target answers the same kind of question as the dataset answer (for example, a company name for a company-name question). A target can be textually distinct but still be invalid for the question.
- `FIRST_SUCCESS_COST`: cumulative actual backward passes, score forwards, generation calls, and wall time at the first qualifying natural-PNG endpoint. Rejected attempts consume budget and remain in these totals.
- `FINAL_ENDPOINT`: the last accepted PNG after the frozen attempt budget, not the best intermediate image. Intermediate hits are recorded but never substituted for the final endpoint.
- `INDEPENDENT_PNG_REPLAY`: PNG evaluated by a separate Python process and fresh model load. A second evaluation inside the attack process is only `IN_PROCESS_REPEAT_REPLAY`.

## Corrections to earlier v27 claims

1. `7006` asks “What is the Company Name?” while the frozen target is “Public Relations Dept.” The target is not a company name. Its prior `TARGET_HIT` is invalid as a valid-answer-type target-control result. The image output may still be recorded as a content change, subject to the semantic endpoint rule.
2. `60700` asks for the subject (`PPBS`) while the frozen target is a person name (`Dr. H. S. SORBS`). Its prior `TARGET_HIT` is invalid as a valid-answer-type target-control result. Do not describe this as evidence that a valid complete wrong answer was induced.
3. `47450` and `47567` have better answer-type matches (depositor/entity and suite number, respectively), but their target hits remain development evidence. The 47450 target's factual wrongness should be confirmed against the source page before using it as a semantic content error.
4. In `run_iterative_compare.py`, `guard_replay()` invokes Full and FastV generation twice in the same Python process. Its flags are `IN_PROCESS_REPEAT_REPLAY`, not an independent-process replay. The v27 results do not establish separate-process PNG evaluation.
5. In v27, equal attempt counts did not always imply equal actual work. In particular, 60700 complete-target used 20 attempts, accepted 10, and consumed 60 backward passes, while complete-front used 10 attempts and 40 backward passes. These are not an equal-compute pair. Report actual costs; do not call the run “all methods equal budget.”
6. On 47450, the first target hit occurred after 14 backwards for Native-FDP, 18 for complete-target, and 16 for complete-front. Thus “front hit in fewer update steps” does not mean it used less compute than Native-FDP. Against complete-target, front used slightly fewer backward passes in this one trajectory; this is exploratory only.

## Current decision status

The old v27 evidence supports continuing only to a small, fixed-order development check. It does not establish that complete-front improves general content failure, and the two strongest earlier target-control examples (7006 and 60700) fail the answer-type audit. No unseen evaluation is authorized by this addendum.

## Integrity note

During a manifest-structure preview, one held-out row (sample 452) was inadvertently printed. It was not attacked or evaluated, but it is now marked exposed and must not be described as untouched. The other held-out rows were not inspected in this turn; all 20 reserved IDs remain excluded from the development sampling rule.
