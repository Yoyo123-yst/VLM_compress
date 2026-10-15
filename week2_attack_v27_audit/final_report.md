# v27 complete-answer comparison and audit

This report separates: (1) real content error, (2) exact hit of a frozen wrong target, and (3) compute and Full-guard behavior.
All data are development data. Unseen samples were not opened.

## Existing six-sample endpoint audit

The detailed table is retained in `existing_result_audit.md`. A target miss is not automatically an attack miss; an answer is judged against the dataset answer and a question-specific rule. Person identity and identifier formatting remain review labels when the string alone is insufficient.

|sample|method|final FastV|content status|target hit|Full exact|time (s)|
|---|---|---|---|---|---|---:|
|47567|native_fdp|2006|CONTENT_FAILURE_NUMERIC|False|True|165.8|
|47567|complete_target|2600|CONTENT_FAILURE_NUMERIC|False|True|183.0|
|47567|complete_front|900|CONTENT_FAILURE_NUMERIC|True|True|135.7|
|50779|native_fdp|Dr. Donbaryx|AMBIGUOUS_PERSON_IDENTITY_OR_SPELLING_REVIEW|False|True|239.6|
|50779|complete_target|Dr. Donbary|AMBIGUOUS_PERSON_IDENTITY_OR_SPELLING_REVIEW|False|True|220.1|
|50779|complete_front|Dr. Dorothy Darby|AMBIGUOUS_PERSON_IDENTITY_OR_SPELLING_REVIEW|True|True|172.8|
|51133|native_fdp|26|CONTENT_FAILURE_NUMERIC|False|True|144.8|
|51133|complete_target|30|CONTENT_FAILURE_NUMERIC|False|True|185.6|
|51133|complete_front|30|CONTENT_FAILURE_NUMERIC|False|True|140.2|
|6194|native_fdp|01017-X00-XX-XX|AMBIGUOUS_IDENTIFIER_FORMAT_REVIEW|False|True|380.5|
|6194|complete_target|01017-X0X-XX|AMBIGUOUS_IDENTIFIER_FORMAT_REVIEW|False|True|383.4|
|6194|complete_front|0107-XX-XX|ANSWER_PRESERVED|False|True|310.7|
|62941|native_fdp|October 13, 1979|CONTENT_FAILURE_DATE|False|True|298.0|
|62941|complete_target|October 13, 1979|CONTENT_FAILURE_DATE|False|True|303.7|
|62941|complete_front|October 13, 1979|CONTENT_FAILURE_DATE|False|True|264.6|
|64506|native_fdp|E|ANSWER_PRESERVED|False|True|286.2|
|64506|complete_target|J|CONTENT_FAILURE_LETTER|False|True|219.7|
|64506|complete_front|C|CONTENT_FAILURE_LETTER|False|True|194.6|

## Fixed-budget reruns

The budget was predeclared as 20 accepted updates for Native-FDP, 13 for complete-target, and 10 for complete-front. This corresponds to about 40 attack/protection backward passes when every attempt is accepted. The report also gives actual counters because the methods have different forward costs.

|sample|method|accepted/attempts|final FastV|first target hit|content status|Full exact|time (s)|score forwards|backwards|
|---|---|---:|---|---:|---|---|---:|---:|---:|
|7006|native_fdp|20/20|Tobacco Company|-|CONTENT_FAILURE|True|315.0|40|40|
|7006|complete_target|13/13|RJ Reynolds Tobacco Company|-|ANSWER_PRESERVED|True|248.7|39|39|
|7006|complete_front|10/10|Public Relations Dept.|8|CONTENT_FAILURE|True|215.5|40|40|
|60700|native_fdp|20/20|PPBS|-|ANSWER_PRESERVED|True|248.6|40|40|
|60700|complete_target|10/20|Dr. H. S. SORBS|10|CONTENT_FAILURE|True|316.8|60|60|
|60700|complete_front|10/10|Dr. H. S. SORBS|5|CONTENT_FAILURE|True|177.0|40|40|
|64506|native_fdp|20/20|E|-|ANSWER_PRESERVED|True|275.3|40|40|
|64506|complete_target|13/13|J|-|CONTENT_FAILURE_LETTER|True|222.8|39|39|
|64506|complete_front|10/10|C|-|CONTENT_FAILURE_LETTER|True|194.4|40|40|

## Findings

1. The complete-front method has a repeatable local signal on 7006 and 60700: it reaches the frozen wrong answer at accepted update 8 and 5, while Native-FDP does not hit that target. On 7006 the final output remains `Public Relations Dept.`; on 60700 it remains `Dr. H. S. SORBS`.
2. This does not prove a general attack-rate gain. Native-FDP also produces clear content changes on several other samples, and exact target hit is a stricter, different objective.
3. 60700 complete-target was rejected 10 times and only accepted 10 updates in 20 attempts. At the same 20-attempt budget it still reaches the target, but its rejection rate is a real weakness and its wall time is higher.
4. 64506 is useful only as a control: `Section 2` is not a valid answer type for a question asking for a letter. The complete methods output `J` or `C`, which are content errors, but no target-hit claim is valid for this frozen target.
5. All accepted checkpoints in the new runs passed Full text exact, Full token exact, Full replay, and the existing PNG perturbation guard. No OOM occurred.

## Semantic review after v27

The six-sample target audit was followed by a source-image and answer-type review. 50779's frozen target `Dr. Dorothy Darby` adds a first name not visible in the clean `To` field, so it cannot be counted as a verified different-person error. 6194 has a dataset answer `0107-XX-XX` but a clean model output `0107-XX-XX-XX`; its digit/mask changes are therefore an identifier-format review case, not a clean content endpoint. 64506's frozen target `Section 2` is invalid for a question whose answer type is a single letter. These cases are excluded from confirmed target-hit claims; details are in `semantic_review.md`.

## Decision

Keep complete-front as a promising local method candidate, but do not open unseen data or claim a general improvement yet. After semantic review, the cleanest evidence is limited to 7006 and 60700. The next scientific step is a pre-registered development holdout of 4--6 samples with valid complete wrong targets and fixed actual compute. If complete-front remains better at comparable actual cost, then and only then consider unseen evaluation. If the advantage disappears, retain it as a targeted-control baseline rather than a new attack method.

The subsequent four-sample confirmation run is summarized in `confirmation_v1_report.md`. It repeats target-control wins on 47450 and 47567, but it is not an untouched holdout and does not establish a general content-failure improvement. 62941 is additionally excluded from strict clean-correct accounting because its clean Full answer already disagrees with the dataset answer.

## Provenance

- Source audit: `existing_result_audit.json` and `existing_result_audit.md`.
- New results: `equal_compute_v27/`.
- Modified runner: `../week2_attack_v25_full_answer_competition_pilot/run_iterative_compare.py`.
- Unseen data: not opened.
