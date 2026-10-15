# v27 confirmation cohort

This is a **confirmation cohort**, not an independent holdout: all four samples were present in earlier development artifacts. No unseen samples were opened. The frozen target manifest, ratio 0.25, PNG guard, selector rule, and compute budgets were unchanged.

Budgets were fixed at 20 accepted updates for Native-FDP, 13 for complete-target, and 10 for complete-front. Every accepted update passed Full text/token exact, Full replay, FastV replay, and uint8 L-infinity <= 8.

|sample|method|accepted/attempts|final FastV|first frozen-target hit|score forwards|backwards|seconds|
|---|---|---:|---|---:|---:|---:|---:|
|47450|Native-FDP|20/20|THE NUTRITION FOUNDATION, INC.|7|40|40|359.3|
|47450|complete-target|13/13|THE NUTRITION FOUNDATION, INC.|6|39|39|297.1|
|47450|complete-front|10/10|THE NUTRITION FOUNDATION, INC.|4|40|40|275.9|
|47567|Native-FDP|20/20|2006|-|40|40|169.8|
|47567|complete-target|13/13|2600|-|39|39|129.2|
|47567|complete-front|10/10|900|7|40|40|115.3|
|51133|Native-FDP|20/20|26|-|40|40|144.2|
|51133|complete-target|13/13|30|-|39|39|123.6|
|51133|complete-front|10/10|30|-|40|40|111.5|
|62941|Native-FDP|20/20|October 13, 1979|-|40|40|308.0|
|62941|complete-target|13/13|October 13, 1979|-|39|39|252.6|
|62941|complete-front|10/10|October 13, 1979|-|40|40|247.7|

## Interpretation

1. The complete-front target-control signal repeats on 47450 and 47567: it reaches the frozen complete wrong answer earlier than Native-FDP, with the same approximate backward-pass budget. On 47450 all three methods eventually reach the target, but front reaches it first. On 47567 only front hits the frozen target in this run.
2. This is not yet a general content-attack improvement. On 51133 and 62941 all methods produce a wrong numeric/date answer, and front does not hit its frozen target. Thus the strongest claim supported here is better control of a pre-registered complete target on some samples, not a higher overall content-failure rate.
3. The shorter front trajectories also finish faster, but wall time alone is not the primary claim because scoring and generation counts differ by method. Backward and score-forward counts are reported explicitly.
4. 62941 is not a strict clean-correct sample: its dataset answer is `December 19, 1979`, while clean Full already outputs `October 13-31, 1979`. It is therefore diagnostic only, not evidence for preserving a correct Full answer.
5. The four samples were previously used during development, so this run cannot be called an untouched holdout. It is a consistency confirmation only.

## Decision

`complete-front` remains worth one carefully frozen development holdout, but the method is not ready for unseen evaluation. Before that holdout, freeze a new set of 4--6 development samples that were not used for target selection or method tuning, and require each target to pass the semantic answer-type check. If such a holdout shows a repeated gain in fixed-selector `TARGET_HIT` or `CONTENT_FAILURE + Full exact` at comparable cost, then run the 20 unseen samples once. If the holdout only repeats target-control wins without improving content failure, report complete-front as a targeted complete-answer baseline rather than a general new attack method.

## Provenance

- Runner: `../week2_attack_v25_full_answer_competition_pilot/run_iterative_compare.py`
- Targets: `../week2_attack_v25_full_answer_competition_pilot/frozen_targets_v26_step3.json`
- Results: `confirmation_v1/`
- Semantic review: `semantic_review.md`
