# v6b soft Full anchor

Decision: `BLOCKED_V6B_CALIBRATION`.

This directory is a separate protocol. The v6 decision `BLOCKED_BC_IMPLEMENTATION_PREREQUISITE` was not changed. No v5b file was written. B and C were not trained, and the 12-sample development run was not started.

## What changed

The v6 hinge `relu(kappa_t - m_adv_t)` is absent. The Full term is

`L_soft_full = mean_t softplus((m_clean_t - m_adv_t) / 1)`

over the clean Full answer tokens and one EOS. Variant C would add the real FastV page-mass L1 term. Coefficients may be taken only from sample 1025 at δ=0, with no probe.

## CPU tests

Nine tests passed, including a nonzero softplus gradient when the adversarial margin equals the clean margin, EOS included once, and rejection of a gradient at or below 1e-6.

## GPU health

One model per free GPU. Started 2026-10-08T18:23:40+08:00. GPU 0 pid 271416 sample 1025, GPU 1 pid 271417 sample 6194, GPU 4 pid 271418 sample 7006. All three processes exited after writing a `health_pass: true` record. GPUs 2, 3, 5, and 7 were left untouched.

| Sample | K | Soft grad L2 | Comp grad L2 | Selection grad L2 | Evidence grad | uint8 reload |
| --- | --- | --- | --- | --- | --- | --- |
| 1025 | 2774 | 4.650 | 9.532 | 0 | 0 | match |
| 6194 | 3699 | 8.842 | 5.877 | 0 | 0 | match |
| 7006 | 3699 | 8.739 | 15.239 | 0 | 0 | match |

On every sample the generate keep count matches the FastV hook, clean Full and FastV text match the stored baseline, and the reloaded uint8 image matches the clean decode with L∞ 0. The soft Full anchor gradient is finite and nonzero. Compression and selection gradients are finite.

## Calibration

The single 1025 measurement has a usable compression gradient and a usable soft-anchor gradient. Their ratio would be about 2.05, but it was not saved as a coefficient. The selection gradient is 0 because the clean and adversarial page masses are the same vector, and the L1 distance is flat there, so beta is undefined. The frozen rule treats a gradient at or below 1e-6 as `BLOCKED_V6B_CALIBRATION` and forbids a probe, a second calibration, or a hand-set coefficient. No smoke and no 12-sample run were started.
