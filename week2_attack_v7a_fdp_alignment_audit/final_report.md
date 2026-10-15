# FDP alignment audit

## Decision

`BLOCKED_MANUAL_FORWARD_MISMATCH`

The v7 decision `NO_GO_NO_COMPRESSION_CONDITIONAL_EFFECT` is unchanged. No PGD was run and B+ was not started. The full 60-sample recompute was not run, because the manual forward does not match generate.

## Paths

G is `model.generate`. T is the cache teacher-forced forward along the clean prefix, using the same processor inputs as generate. M is `full_answer_logits`.

On all 10 audited image conditions, G and T choose the same token at every shared-prefix step. Two greedy generates also match each other. The largest absolute logit gap between G and T is 0.0625.

M does not match T. The largest absolute logit gap is 0.5. Argmax flips in three conditions:

- 60744 random noise, step 0. Clean token 68781, G and T choose 46616, M chooses 68781. T's hard margin on the clean token is 0. The decoded texts are `SIMON FRASER UNIVERSITY` and `Simon Fraser University`.
- 1023 clean, step 3. G and T choose 17 with margin 0.125. M chooses 16 with margin 0.
- 51140 random noise, step 0. G and T choose 27160 with margin 0.125. M chooses 41 with margin 0.

The generated text still matches on 1023 and 51140 because generate follows T, not M.

## Corrected certificate logic

The accepted implication is one way: if every real generated token has hard margin above `epsilon_margin`, the generated token ids equal the clean token ids. The old check `full_exact == cone_holds` is not used.

Under T, 60744 noise has cone false, so it is not a certificate violation. It is a tie at the first token, and greedy decode selects the other casing. The old violation was M assigning a positive margin to the clean token.

56084 clean has a first-step margin of exactly 0 while the tokens still match. That is `CERTIFICATE_UNAVAILABLE_LOW_MARGIN`, not a violation.

## 22317

All three images produce 16 tokens and stop at `max_new_tokens`. The last token is 19050, not EOS 151645. The termination is `MAX_LENGTH_TERMINATION`. The old 17th margin, whose value was about -18, was an appended EOS and is not part of the certificate. On the real 16 tokens the cone holds and the tokens match.

## Old gate D

The six old conflict ids are 5615, 1023, 22317, 51140, 56084, and 60744. On this audit they are false positives of the two-way check, the appended EOS, or the manual forward. None is a true certificate violation under G and T. The other 54 samples were not remeasured.

## What FDP can be

`full_answer_logits` cannot be the certificate or the FDP loss until it uses the same forward as generate. With the current manual forward, FDP is not a strict guard for a future B+. It would only be an ordinary margin regularizer, and it can point at the wrong token when the true margin is 0.125. The minimal repair is to take the logits from the native cached forward, score only tokens the model actually generated, and keep the one-way cone test. That repair was not implemented here.

The existing Variant B strict result, 52/60, is outside this audit.
