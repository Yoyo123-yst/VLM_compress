# Frozen target review (before attack)

The six-ID cohort was frozen first. Four samples failed clean admission; only `4742` and `61240` proceed. Candidate generation used clean FastV only, capped at 24 raw branches and 6 complete continuations. Dataset answers and source pages were consulted only after candidate generation for this semantic review.

## 4742

- Question: “Where is the Delivery Point?” Dataset answer: `Morehead, Ky`.
- Clean source evidence page explicitly shows “Delivery Point Morehead, KY”; clean FastV says `Morehead Ky` and clean Full says `Marehead Ky` (ANLS 0.909).
- Candidate 0 `MOOREHEAD KY`: spelling variant of Morehead; reject as wrong target.
- Candidate 1 `Morehead Kentucky`: answer alias/format variant; reject.
- Candidate 2 `Marshall, KY`: a complete location of the requested type, distinct from the answer and not a spelling/format variant. The source line names Morehead, not Marshall. **Freeze this first valid candidate**.
- Remaining candidates are not reached by the first-valid rule.

## 61240

- Question: “Who is the National foundation state representative of Connecticut?” Dataset answer: `Mrs. Ruth Shiffrin`.
- Clean source evidence page explicitly states “National Foundation State Representative - Mrs. Ruth Shiffrin”; clean FastV says `Ruth Shiffrin` and clean Full also says `Ruth Shiffrin` (ANLS 0.765).
- Candidate 0 `Mrs. Ruth Shiffrin`: exact answer alias; reject.
- Candidate 1 `MR. RUTH SHIFFRIN`: title/gender change with the same surname; identity is not safely established as a different person; reject as ambiguous.
- Candidate 2 `Mary E. Fordman`: complete person name, answer type matches, and it differs from the source-named representative. **Freeze this first valid candidate**.
- Candidates 3–5 are title/spelling variants of Ruth Shiffrin and are rejected.

No target was manually invented. The target for each sample is copied from the generator's frozen clean-only continuation. The exact same target tokens and correct-expression list are used for `complete_target` and `complete_front`.

## Scope

These are two construction samples only. Four of the six frozen IDs remain in the cohort denominator as `CLEAN_GATE_FAILED`; they are not replaced. The two valid targets permit a small paired pilot, not a reliable estimate of population success or a generalization claim.
