# Execution addendum

This file locks evaluation rules that were already specified for v5b. It does not change λ, steps, ε, the loss, the 40 sample IDs, or `FROZEN.json`.

- Primary result is the disk image at step 200.
- Min-margin is a pre-registered secondary selector and reads only validity, compressed margin, and step.
- Oracle best-strict is diagnostic only.
- Random noise is one Rademacher ±8 draw per sample from seed `2026100804`.
- Pass requires 27/40 strict successes, Wilson lower bound above 0.5, no Full errors, unchanged Full text, full replay, and McNemar p < 0.05 against noise.
- Engineering smoke on sample 50743 is not part of the 40.
