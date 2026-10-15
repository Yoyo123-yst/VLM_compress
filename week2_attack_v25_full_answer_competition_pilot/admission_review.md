# Full-answer competition pilot: clean candidate admission

This is a development-only, clean-input screening artifact. No attack PNG,
pixel update, or unseen sample was used.

## Decision

`GO_TWO_SAMPLE_COMPLETE_TARGET_SCORE_SMOKE`

The first candidate generator incorrectly reused v13's local branch probe: it
kept one competing token per position and truncated the global pool before
seeing complete continuations. That produced mostly case changes, titles, and
misspellings, so its earlier candidate-level NO-GO was not informative about
complete-answer targeting.

Version 2 expanded the clean FastV branch pool and completed each branch before
screening. It exposed a second defect: its differentiable `pixels_from_page`
path did not reproduce the processor tensor or natural FastV keep-set at zero
perturbation.

Version 3 fixes this with a processor-anchored differentiable tile delta. For
both admitted samples, the zero-injection tensor, FastV tokens, K, and keep-set
are exact. The complete candidates below survive that correction.

## Frozen development targets

### 47450

- Question: `Who has made the deposit ?`
- Correct answer group: `Dr. or Mrs. William J. Darby`,
  `Dr. Or Mrs. William J Darby`, `William J Darby`.
- Frozen wrong target: `THE NUTRITION FOUNDATION, INC.`
- Source: clean FastV position-0 rank-3 branch followed by greedy conditional
  completion.
- Semantic review: complete valid output, wrong entity and wrong answer type;
  no correct alias is preserved.
- Corrected clean target-to-competitor gap: 3.375.

### 7006

- Question: `What is the Company Name ?`
- Correct answer group: `RJReynolds Tobacco Company` and normalized spacing/
  case variants.
- Frozen wrong target: `Public Relations Dept.`
- Source: clean FastV position-0 rank-3 branch followed by greedy conditional
  completion.
- Semantic review: complete valid output, a department rather than the named
  company; no correct alias is preserved.
- Corrected clean target-to-competitor gap: 2.4375.

Candidate proposal and ranking did not use dataset answers or ANLS. Dataset
answers were used only after generation to exclude correct aliases and confirm
that the frozen targets are semantically wrong. Forced continuation remains a
screening operation and is not an attack endpoint.

## Sequence-score smoke result

`PASS_TWO_SAMPLE_SCORE_DIRECTION_AND_PNG_GATE`

At uint8 L-infinity 1, both samples move in the intended direction after an
independent PNG processor replay, while Full generation remains token exact:

- 47450: grouped objective 0.9852 -> 0.8418 on PNG replay; wrong-target
  advantage versus the best correct expression -0.2928 -> -0.2037.
- 7006: grouped objective 1.2078 -> 1.0946 on PNG replay; wrong-target
  advantage versus the best correct expression -0.7349 -> -0.6213.

The natural FastV answer remains correct after this single step. This smoke
validates the optimization direction and execution path; it is not attack
success.

## Next gate

The next permitted experiment is a small iterative comparison on the same two
development samples:

1. Native-FDP;
2. ordinary complete-target loss against the single clean FastV sequence;
3. the same complete wrong target against the frozen correct-alias group.

All three must use the same accepted-update budget, real Full exact guard,
pixel/optimizer rollback, checkpoint selector, and PNG replay. The grouped
method earns a discriminator witness only if it succeeds where the ordinary
complete-target baseline fails, or reaches the same legal content failure at
lower measured cost.

