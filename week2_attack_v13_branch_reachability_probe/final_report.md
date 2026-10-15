# Compression-conditioned branch reachability probe

Status: `BRANCH_CONSTRUCTION_SUPPORTED_SELECTOR_ADVANTAGE_NOT_ESTABLISHED`.

This is a small development construction probe. It establishes that meaningful
FastV answer branches can be targeted from the auxiliary page while the Full
output remains exact. It does not establish that the proposed Full-aware
reachability score improves attacks, and it is not an unseen result.

## Frozen setting

- Target and threat model: the existing source-bounded white-box
  LLaVA-OneVision/FastV path; only the designated auxiliary rendered PNG page
  is optimized; `L∞ <= 8/255`.
- Training ratio: `r=0.25`; 25 Adam updates; `lambda_full=2.6371882367154997`.
- Every update recomputed the current FastV keep set and then passed a two-call
  real Full generation guard. A rejected update would restore both `delta` and
  Adam state.
- Three equal update-budget variants were compared on 60700:
  `SEQ` attacks the fixed clean FastV sequence; `NAIVE` picks the candidate
  with the smallest estimated FastV flip radius; `AWARE` picks the candidate
  with largest estimated Full local capacity divided by that radius.
- Candidate screening used lexical content filtering and forced continuation
  only to remove punctuation/case-equivalent branches. It never used answer
  labels or ANLS. Success always used an unforced saved PNG.

## Admission and selected branches

Both initially requested samples passed the clean Full and clean FastV gates.
47450 was not run through all three variants because its naive and aware
selectors chose the same `Dr -> William` branch; it provides no test of the
selection rule.

On 60700, clean Full and FastV both returned `PPBS`.

| selector | branch | estimated FastV flip radius | estimated Full capacity | capacity/radius |
|---|---|---:|---:|---:|
| NAIVE | `PP -> Pages` | 0.000303 | 0.033333 | 109.88 |
| AWARE | `BS -> SB` | 0.000456 | 0.066667 | 146.23 |

The conditioned completions were `Pages 3 and 4` and `PPSB`, respectively.
They are diagnostics of branch semantics, not endpoint outputs.

## Real endpoint trajectory

| variant | first strict checkpoint | unforced FastV text | Full text | fresh-process replay |
|---|---:|---|---|---|
| SEQ | 10 | `Ppbs #14, 1967` | `PPBS` | pass |
| NAIVE | 10 | `Pages 3 and 4` | `PPBS` | pass |
| AWARE | 15 | `PPSB` | `PPBS` | pass |

All three first-success PNGs passed fresh model reload, two Full generations,
two FastV generations, Full token/text exactness, valid output, and uint8
bound checks. Thus fixed branch targeting is physically realizable through
the actual FastV path.

The current Full-aware score did not predict the faster branch: it chose the
higher ratio `BS -> SB` branch, but that branch reached strict failure at step
15, five updates after `SEQ` and `NAIVE`. It later returned to correct FastV
output `PPBS` at step 25. The selector based on the lowest fixed pair margin
would choose AWARE step 20, which also passed fresh replay, but that is
checkpoint selection rather than evidence that the update rule maintains an
error state.

## Interpretation and next causal check

The recovery is mechanically expected from the present branch hinge loss.
Once `z(y_t)-z(c)` is below zero, `max(0, z(y_t)-z(c))` has zero attack
gradient. The remaining Full-FDP term can still move the pixels, so a reached
branch can drift back. This is a concrete defect to test, not a story about
geometry.

The next bounded ablation is therefore an AWARE state-retention check on this
same sample: after the fixed branch margin first reaches the target while the
Full guard passes, freeze the legal best branch state and verify at step 25
that the saved PNG still naturally generates the same FastV error. If this
does not remove the recovery, stop this direction. If it does, the following
question is whether state-gated allocation across multiple compression
budgets adds value beyond a fixed best-checkpoint selector. No six-sample or
unseen run is justified before that check.
