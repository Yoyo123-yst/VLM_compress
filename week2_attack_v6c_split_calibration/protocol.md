# v6c split calibration

This directory is a new protocol. v6 remains `BLOCKED_BC_IMPLEMENTATION_PREREQUISITE`. v6b remains `BLOCKED_V6B_CALIBRATION`. Those decisions are not reopened.

Variant B uses `lambda_soft = 2.049824826423478`, the exact quotient of the immutable v6b health record on sample 1025. B does not use beta.

Variant C uses that same lambda and one pre-registered probe: add `1/255` to the auxiliary page of sample 1025, clamp to `[0, 1]`, and set `beta` to the ratio of the compression and selection gradient L2 norms at that single state. A zero selection gradient at the clean image is expected and is not a health failure. If the probe selection gradient is not finite or is at most `1e-6`, C is `C_BLOCKED_SELECTION_PROBE_CALIBRATION` and B continues.

The development ids, FastV ratio, model, auxiliary-only permission, epsilon, Adam step size, 200 steps, checkpoints, and prospective selector match the earlier development protocol. Matched noise for the later 40-sample check is the existing v5b noise file, read only.
