# Core-content objective baseline

## Scope

This is a two-sample diagnostic, not a method result. The only change from
the repaired v19 protocol is that the attack surrogate averages the soft
margin over the clean FastV answer span; Full-FDP, exact Full replay, PNG
quantization and the natural/frozen selection modes are unchanged.

## Results

Four trajectories were run in parallel:

- 60700 natural and frozen;
- 7006 natural and frozen.

Each trajectory used four step sizes and four updates. All `80/80` records
passed Full exact/replay and the uint8 perturbation bound.

### 60700

The clean core span was `PPBS`. The content objective reduced the core margin
substantially. Final outputs included `PPSB`:

- natural: `2/4` final settings had strict ANLS failure;
- frozen: `3/4` final settings had strict ANLS failure.

`PPSB` is a core-token permutation, unlike the earlier `PPBS 14, 1967`
suffix artifact. This is stronger evidence of a content-level transition, but
it is still one development sample and does not establish generality.

### 7006

The clean reference span was conservatively taken as all five non-EOS tokens
of `RJ Reynolds Tobacco Company`. The content objective did not produce a
strict endpoint in either mode. Core margins stayed flat or increased, and
outputs alternated between `Tobacco Company` and `RJ Reynolds Tobacco
Company`.

## Decision

The core-content objective is a promising stronger baseline for the 60700
failure mode, but it does not solve the 7006 recovery mode. The next valid
experiment is a small semantic/content-loss ablation on additional development
samples, with `PPSB` manually or rule-based verified as a genuine content
error. No unseen evaluation or dynamic multi-budget mechanism should be
started yet.
