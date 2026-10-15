# v6 development addendum

This file does not replace `FROZEN_DEV_PROTOCOL.json`. It freezes the Variant A audit rule and the B/C losses before either B or C is trained.

Parent SHA-256: `2e9a05f5d1be778de6321f783a99c790d85f75eae37e17e1370805fb66de0c87`.

v5b raw hashes from `mechanism_audit_v1/raw_before.sha256` were recomputed: 212 files, 0 changes.

Nonzero perturbation is the maximum absolute uint8 pixel gap, not a PNG-file hash comparison. The selector never reads ANLS, FastV text, or replay. Replay happens after the pick and a failed replay is not replaced by another checkpoint.

Variant B replaces Full KL with a token-wise hinge on the clean Full answer, including EOS. Kappa is `max(0, min(clean margin, 1))` from the clean image only. Variant C adds one L1 penalty on page-level FastV attention mass. Lambda and beta are one calibration on sample 1025 at delta 0, shared by all 12 development samples.
