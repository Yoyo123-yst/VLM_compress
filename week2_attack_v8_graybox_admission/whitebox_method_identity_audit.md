# White-box method identity

Verdict: `MATERIALLY_DIFFERENT`.

Source of the executed objectives: `week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation/run_v7d.py` `backward_method`, `anchor_grad`, `fdp_grad`, `fast_grad`, and `losses.py`. Coefficients are the frozen file `FROZEN_COEFFICIENTS.json`, not a restatement from the report prose.

## Shared pieces

Both methods, as actually stepped, add the same compressed term before their Full term.

Let `y_fast(r)` be the tokens returned by `production_generate` under the installed FastV hook at ratio `r`. No token is appended. If generation stops at `max_new_tokens=16` without EOS, the last id is that real token.

`m_fast(r) = mean_t soft_margin(z_t, y_fast(r)_t)`

`L_compressed = 0.5 * (m_fast(0.25) + m_fast(0.35))`

`soft_margin(z, y) = z_y - logsumexp_{v ≠ y} z_v`

The FastV indices are taken from the live hook and then replayed by `fastv_rows`. That path is target FastV. It is why both methods are white-box.

Clean Full tokens `y_clean` come from unhooked `model.generate` (`do_sample=False`, `max_new_tokens=16`, `use_cache=True`). `native_rows_from_full_ids` concatenates the prompt with exactly those ids and keeps predictor positions `prompt_len-1` through `prompt_len-1+T-1`. It does not append EOS.

Clean hard margins `a_t` are detached. The decision floor is

`a_bar_t = min(a_t, max(a_t - ξ, η))`

with `ξ = 1` and `η = 0.001`. `a_bar_t` cannot exceed `a_t`.

Shared step rule: Adam, learning rate `2/255`, `ε = 8/255`, 200 steps, seed `2026101102` reset per sample, uint8 projection after each step. A non-finite leaf gradient raises and stops that step. Gradients are one term per backward. The leaf is the auxiliary delta.

`λ_full = 2.6371882367154997`, taken from the +1/255 probe on sample 51070.

## B_NATIVE

`L_B_NATIVE = L_compressed + λ_full * L_anchor`

`L_anchor = mean_t softplus(a_bar_t - s_t)`

`s_t` is the soft margin of the fixed clean token `y_clean_t` on the attacked Full native-cache row. `anchor_grad` passes `check_argmax=False`. The floor is used. The attacked path's own greedy tokens are not the teacher.

## NATIVE_FDP

`L_NATIVE_FDP = L_compressed + λ_full * L_FDP`

`L_FDP = mean_t softplus(a_t - h_t)`

`h_t` is the hard margin of the same fixed clean token: chosen logit minus the max rival. `fdp_grad` also teacher-forces `y_clean` and does not check argmax. It does not use `a_bar`, soft margin, or the B+ gap.

This is one-sided. It penalizes the attacked hard margin falling below the clean hard margin. It is not a two-sided claim that the Full argmax string stays exactly the clean string.

## Why they are not the same function

The Full terms differ in three executed ways: soft margin versus hard margin, a lowered floor versus the raw clean hard margin, and `soft_anchor` versus `decision_preservation`. `tests/test_losses_cpu.py` evaluates the two method losses on the same dummy tensors and requires the floats to differ. `backward_method` calls `anchor_grad` only for `B-Native` and `fdp_grad` only for `native-FDP` and `B+`.

Native-FDP's Full term uses only the real Full native-cache rows for the real clean generate tokens, and it does not add a fake EOS. The whole Native-FDP objective is still white-box, because `L_compressed` reads target FastV tokens, keep indices, and FastV gradients.

B_PLUS is not re-derived here. Its executed extra term is `-λ_gap * D` with `D = m_full - L_compressed`, and the development result already records higher Delta with lower strict. That ablation stays as written.
