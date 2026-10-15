# Iterative complete-answer competition pilot (v2)

## Scope

This is a two-sample development pilot on samples `47450` and `7006`, ratio
`r=0.25`, with five accepted sign-PGD updates per method. It is not an unseen
result and does not open the test set.

Methods were run in fresh processes with the same model, processor, anchored
pixel mapping, uint8 PNG quantization, `L_inf <= 8/255` guard, and independent
Full replay. The three attack objectives were:

1. `native_fdp`: the existing dynamic FastV self-margin objective plus
   Full-FDP;
2. `complete_target`: one frozen complete wrong-answer sequence against the
   clean FastV sequence plus Full-FDP;
3. `alias_group`: the frozen wrong-answer sequence against a log-sum-exp group
   of registered correct aliases plus Full-FDP.

The model is frozen; only the auxiliary-page pixels receive gradients. No
`generate`, `backward`, or forward-hook wrappers are installed.

## Protocol correction

The first v1 pilot is invalid and must not be used as evidence. Its update
routine accidentally computed every gradient at the zero-perturbation page,
so the logged attack loss was constant across updates. It had no replay or
Full-exact violation, but it did not implement iterative optimization. v1 is
retained only as a failed implementation audit.

v2 passes a current `delta` into the objective. A smoke check showed the
Native-FDP attack loss changing from `4.389217` to `4.222830` after the first
accepted update, confirming that the update is evaluated at the current state.

## Results

| sample | method | accepted/attempts | attack loss first -> last | FastV natural output | Full exact/replay |
|---|---:|---:|---:|---|---|
|47450|Native-FDP|5/5|4.389217 -> 3.553421|unchanged `Dr. or Mrs. William J. Darby`|pass for every update|
|47450|complete-target|5/5|0.292765 -> 0.043247|unchanged `Dr. or Mrs. William J. Darby`|pass for every update|
|47450|alias-group|5/5|0.985212 -> 0.595248|unchanged `Dr. or Mrs. William J. Darby`|pass for every update|
|7006|Native-FDP|5/5|3.059503 -> 2.740527|unchanged `RJ Reynolds Tobacco Company`|pass for every update|
|7006|complete-target|5/5|0.734856 -> 0.461378|unchanged `RJ Reynolds Tobacco Company`|pass for every update|
|7006|alias-group|5/5|1.207771 -> 0.880058|unchanged `RJ Reynolds Tobacco Company`|pass for every update|

All accepted PNGs had actual uint8 `L_inf` values 1, 2, 3, 4, and 5 in order.
No update was accepted without a changed PNG. No Full text/token exact or
replay failure occurred.

## Interpretation

The complete-answer objective is implementable and its differentiable proxy
decreases substantially, especially on `47450`. However, neither complete
targeting nor alias-group targeting produced a natural FastV content failure
within the tested budget, and neither has a discriminator witness over
Native-FDP. This is a feasible objective pilot, not a successful new attack.

The next gate, if pursued, is a pre-registered extension to at most 10--25
accepted updates on the same two development samples, with independent PNG
evaluation and a fixed selector. It should stop if the natural answer remains
unchanged while only the proxy improves. No unseen evaluation is authorized
from this pilot.

Artifacts are under `iterative_v2/`; each method/sample has a `result.json`
and accepted PNGs. The v1 directory is explicitly invalid due to the stale
page-gradient bug.
