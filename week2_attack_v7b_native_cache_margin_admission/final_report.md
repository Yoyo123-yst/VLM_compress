# Native-cache margin admission

Decision: `PASS_NATIVE_CACHE_MARGIN_ADMITTED`.

This stage did not recompute or rewrite Variant B's existing 52/60 strict `model.generate()` result. It does not give B a retrospective theoretical certificate. It only decides that the native-cache margin is a legal basis for a future B+ and for future mechanism analysis.

## What was checked

CPU tests in `tests/test_native_margin_cpu.py` passed (6/6): truncation does not append EOS, EOS is scored only when generated, `cone_holds` is not treated as equivalent to `full_exact`, the competitor set excludes the target token, the native module does not call `full_answer_logits`, and a missing or mismatched FastV K / keep hash fails.

H1 Full generate vs native cache replay: 10/10. Text, token ids, and every-step argmax match. The largest absolute logit gap on the shared prefix is 0.03125. That gap is reported only; it was not used as a threshold.

H2 FastV generate vs FastV native replay at keep ratio 0.25: 10/10. Token ids, text, every-step argmax, K, and keep hash match. The largest absolute logit gap is 0.03125. The keep set is the one recorded by the real FastV hook.

H3 native soft-margin gradients on the first three frozen conditions:

| condition | aux L2 | evidence L2 | cache argmax match |
| --- | ---: | ---: | --- |
| 60744 random_noise | 11.268 | 0 | yes |
| 60744 clean | 99.909 | 0 | yes |
| 1023 clean | 11.384 | 0 | yes |

The gradient is the mean soft margin of the real generated tokens through `model()`. Per-layer recomputation is enabled only for that backward so the graph fits in memory. Dropout is 0, so this does not change the forward values. No new lambda was chosen from these L2 values.

## Records that are not failures

`60744 / random_noise` generates `SIMON FRASER UNIVERSITY`, which is not the clean text, and its hard margin is not positive on every real token. `cone_holds` is false and `full_exact` is false. Generate and the native replay still choose the same tokens. Status: `CERTIFICATE_UNAVAILABLE`.

`22317` on clean, B, and random noise stops at 16 real tokens with `MAX_NEW_TOKENS_TRUNCATED`. The last token is not EOS. No margin was computed for an appended EOS.

`cone_holds` means every real generated token has hard margin > 0. It is a one-way sufficient condition. It is not compared with `full_exact` by equality.

## Provenance

The seven protected directories still have the same 3255 file hashes as `raw_immutability_before.json`. `raw_immutability_check.json` is pass. No PGD, B+ training, new cohort, or 60-sample native recompute was run.
