# Native60 readonly remeasure

Decision: `PASS_NATIVE60_SWEET_SPOT_AND_DIFFERENTIAL_SUPPORTED`.

Variant B's existing r=0.25 strict result remains 52/60 from the original generate endpoint. This stage did not recompute, replace, or weaken that result. Native-cache margin is a differentiable proxy for mechanism analysis and a possible future B+. It is not the success endpoint.

All 600 Full and FastV native paths matched real generate on tokens, argmax, and decoded text. FastV K and keep hash matched the real hook on every FastV path. No pixels, checkpoints, or selectors were changed.

Parent raw immutability: PASS.

Remeasured strict rates, from this run's generate texts under the frozen predicate:

- r=0.5: B 11/60, noise 0/60, Delta mean 1.8818, 95% CI (1.4152, 2.3210).
- r=0.35: B 48/60, noise 0/60, Delta mean 2.7128, 95% CI (2.1523, 3.2250).
- r=0.25: B 52/60, noise 1/60, Delta mean 2.3898, 95% CI (1.8034, 2.9332).
- r=0.15: B 45/60, noise 13/60, Delta mean 2.2273, 95% CI (1.6828, 2.7582).

Sweet-band mean Delta is 2.5513. Outer-band mean Delta is 2.0546. Their paired difference is 0.4967, with 95% CI (0.1024, 0.8880). The exact sign test on the non-zero paired differences has p=0.366. The mean comparison clears its preregistered bootstrap bound. The sign test does not.

A positive Delta means B opens a larger Full-versus-FastV margin gap than the paired random image at the same budget. It does not mean the generate endpoint is a strict success.

This result applies to LLaVA-OneVision with FastV on MP-DocVQA under the current auxiliary-page threat model. It is not a universal claim and it does not prove that Full exact is theoretically guaranteed. B+ was not started.
