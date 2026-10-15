# Representation intervention probe

## Question

Does the attack survive because the FastV keep-set changes, or because the
perturbed auxiliary page carries a different representation into the decoder?

## Protocol

Four existing development PNGs were replayed: `1025`, `2008`, `7006`, and
`39196`. For each sample, Full and FastV were first replayed on clean and
attack images. At the input of the real FastV layer-3 pruning hook, the probe
then performed two interventions:

1. **Repair:** replace one page's attack hidden states with the corresponding
   clean hidden states, then run natural FastV generation.
2. **Reverse:** inject attack hidden states for one page into the clean image,
   then run natural FastV generation.

FastV's keep-set is computed at layer 2 before this intervention, so the
intervention does not change the pruning decision. No forced output or token
target was used.

## Replay results

All four attack images preserved the clean Full output exactly. The natural
FastV outputs were:

| sample | clean FastV | attack FastV | repair auxiliary page | repair evidence/other pages | reverse auxiliary page |
|---|---|---|---|---|---|
| 1025 | `Dr. C. Baur` | `Tanis Tanemar` | `Dr. C. Baur` | `Tanis Tanemar` | `Tanis Tanis` |
| 2008 | `March 1956` | `February 23, 1966` | `March 1956` | `February 23, 1966` | `February 23, 1966` |
| 7006 | `RJ Reynolds Tobacco Company` | `Tobacco Company` | `RJ Reynolds Tobacco Company` | `Tobacco Company` | `Tobacco Company` |
| 39196 | `Technical Development Manager` | `Chaplain, New Jersey` | `Technical Development Manager` | `Chaplain, New Jersey` | `Chemical Rubber Co.` |

For 1025, 2008, and 7006, replacing only the auxiliary-page representation
restored the clean FastV output, while replacing evidence or other-page
representations did not. The reverse intervention reproduced the attack
answer family on all four samples, although it need not reproduce the exact
PNG endpoint.

## Interpretation

The repair and reverse interventions show that the auxiliary-page hidden
states are a carrier of the FastV-sensitive effect, but the first random
control was insufficient. It was corrected to start from the attack hidden
state and add a random direction with the same Frobenius norm as the
attack-to-clean displacement.

With this matched control, the random intervention produced:

| sample | matched random auxiliary intervention |
|---|---|
| 1025 | `Dr. Czaba` (near-neighbor, not clean)
| 2008 | `March 1956` (clean FastV answer)
| 7006 | `RJ Reynolds Tobacco Company` (clean FastV answer)
| 39196 | `Technical Development Manager` (clean FastV answer)

The follow-up directional controls were run from the attack hidden state with
the same Frobenius norm as the attack-to-clean displacement. The exact clean
displacement recovered the clean FastV answer on 4/4 samples. Random controls
recovered it on 3/4 samples (all three seeds on 2008, 7006, and 39196), while
orthogonal controls recovered it on 2/4 samples (all three seeds on 7006 and
39196). On 1025, both controls also produced near-boundary alternatives such
as `Dr. C. Baurer` and `Dr. Czaba`; on 2008, orthogonal controls produced
`March 1966`/`March 1968`. Thus the exact direction has a weak directional
advantage, but random and orthogonal movements frequently cross the same local
decoder boundary. This is not yet an independent mechanism.

The supported part remains:

- the attack and clean images have the same page layout and the same real
  FastV pruning rule;
- the Full attack output remains exact;
- changing only the auxiliary page hidden state before pruning repairs the
  compressed answer;
- injecting that hidden state into the clean run induces an error-family
  output.

The supported mechanism is therefore:

```text
auxiliary-page pixels
  -> auxiliary-page pre-pruning hidden representation
  -> compressed decoder decision
  -> FastV answer change
```

This does not prove that the effect is independent of all later-layer
cross-token interactions; the controls below explicitly test that possibility.

## Decision

`NO_GO_AUX_DIRECTIONAL_MECHANISM`.

This is a useful localization result, but it is not yet an independent causal
witness for a new attack mechanism. Do not start method optimization or claim a
new algorithm from it.

## Completed directional and layer-wise checks

The next probe must distinguish directional semantics from generic boundary
sensitivity:

1. Construct three controls from the attack hidden state: the exact clean
   displacement, a random displacement with the same norm, and a displacement
   orthogonal to the clean displacement with the same norm.
2. Repeat each control with at least three fixed random seeds.
3. The current result passes only the weak frequency check (4/4 versus 3/4
   versus 2/4), not a strong uniqueness check. Do not start attack-method
   optimization.
4. The layer-wise follow-up was run. Auxiliary repair produced:

   | sample | layer 1 | layer 2 | layer 3 |
   |---|---|---|---|
   | 1025 | `Dr. Bacharach` | `Dr. Bacharach` | `Dr. C. Baur` |
   | 2008 | `March 1956` | `March 1956` | `March 1956` |
   | 7006 | `RJ Reynolds Tobacco Company` | `RJ Reynolds Tobacco Company` | `RJ Reynolds Tobacco Company` |
   | 39196 | `Technical Development Manager` | `Technical Development Manager` | `Technical Development Manager` |

   Thus only 1025 localizes to the pre-pruning layer-3 boundary; the other
   three are repaired by earlier hidden-state replacement as well. Combined
   with the frequent matched-random recoveries, this is not a stable,
   layer-specific causal mechanism.
5. Stop this representation-direction route. Do not start a new attack
   objective, cross-budget scheduler, or unseen evaluation from it.

The layer-wise artifact is `layerwise_intervention.json`. The result supports
only a representation-sensitive error localization: auxiliary-page hidden
states mediate the observed endpoints, but no independent directional or
layer-specific attack mechanism has been established.

The clean displacement wins only weakly in frequency and no universal layer is
localized. The finding therefore remains an error analysis, not a new attack
objective.

Artifact: `representation_intervention.json`.
