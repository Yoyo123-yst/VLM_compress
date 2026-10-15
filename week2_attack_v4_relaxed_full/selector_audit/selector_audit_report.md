# Checkpoint 选择口径审计

## 1. 结论摘要

- 原始 holdout checkpoint 记录完整性：**PASS**。
- **主 endpoint** 应为 `primary_fixed_step200`（每题固定 step 200），不是 `best_strict`。
- λ=0.25 主 endpoint strict CSFR = **15/20**（0.750），Full 错误率 0.000。
- post-hoc min-margin：**16/20**。
- oracle best_strict 上界：**17/20**（`GROUND_TRUTH_ORACLE_UPPER_BOUND`）。
- λ=0.25 相对 λ=1 的额外优势：点估计更高，但 n=20 配对区间包含 0，**不能**写成显著优于。

## 2. 原始记录完整性

见 `raw_integrity.md`。两种方法各 20 题、ID 与 frozen_manifest 一致，每题 8 个 checkpoint（0,10,25,50,75,100,150,200），必填字段齐全。

## 3. 为什么 best_strict 不是主 endpoint

`select_best` 先保留 Full ANLS≥0.5，再最小化 FastV ANLS，tie-break 为 margin、linf、step。选择时使用了真实 Full/FastV 表现，因此是 **ground-truth oracle**，只能作诊断上界。

## 4. 三种 selector 定义

| 名称 | 选择规则 | 选图时是否用 ANLS/答案 |
| --- | --- | --- |
| primary_fixed_step200 | step=200 | 否 |
| posthoc_min_compressed_margin | output_valid 内最小 margin，同 margin 取更早 step | 否 |
| diagnostic_oracle_best_strict | 复现 `loss.select_best` | 是 |

## 5. λ=0.25 三种口径

| selector | strict CSFR | Full error | FastV error | mean G | 选图用答案/ANLS |
| --- | ---: | ---: | ---: | ---: | --- |
| primary_fixed_step200 | 15/20 | 0.000 | 0.750 | 0.650 | 否 |
| posthoc_min_compressed_margin | 16/20 | 0.000 | 0.800 | 0.700 | 否 |
| diagnostic_oracle_best_strict | 17/20 | 0.000 | 0.850 | 0.733 | 是 |

λ=0.25 fixed step 200：Full 文本与 step0 一致 **20/20**；Full ANLS 与 step0 一致 **20/20**。

## 6. λ=1 三种口径

| selector | strict CSFR | Full error | FastV error | mean G |
| --- | ---: | ---: | ---: | ---: |
| primary_fixed_step200 | 12/20 | 0.050 | 0.650 | 0.522 |
| posthoc_min_compressed_margin | 12/20 | 0.050 | 0.650 | 0.522 |
| diagnostic_oracle_best_strict | 14/20 | 0.000 | 0.700 | 0.629 |

oracle 口径下 Full error=0 来自选择器先过滤 Full ANLS≥0.5 → **FULL_SAFETY_IS_SELECTOR_CONDITIONED**。

## 7. 配对 λ 比较（按 selector 分开）

- **primary_fixed_step200**：candidate-only 3, baseline-only 0, McNemar p=0.25, sign-flip p=0.125 (nonzero pairs 6), mean ΔG=0.1281, t-CI95 [-0.0234, 0.2795], bootstrap-CI95 [0.0039, 0.2797], λ advantage=NOT_ESTABLISHED.
- **posthoc_min_compressed_margin**：candidate-only 4, baseline-only 0, McNemar p=0.125, sign-flip p=0.0625 (nonzero pairs 6), mean ΔG=0.1781, t-CI95 [0.0201, 0.3360], bootstrap-CI95 [0.0476, 0.3318], λ advantage=NOT_ESTABLISHED.
- **diagnostic_oracle_best_strict**：candidate-only 3, baseline-only 0, McNemar p=0.25, sign-flip p=0.25 (nonzero pairs 3), mean ΔG=0.1039, t-CI95 [-0.0157, 0.2236], bootstrap-CI95 [0.0000, 0.2227], λ advantage=NOT_ESTABLISHED.

在固定 step 200 主 endpoint 上：mean ΔG 的 t 区间包含 0，而 percentile bootstrap 区间不包含 0；但精确 McNemar 与精确 sign-flip 均未拒绝原假设。不同推断方法边界不一致，n=20 不足以证明 λ=0.25 稳定优于 λ=1。

## 8. Full 保护分析

- fixed step 200（λ=0.25）：Full 文本不变 20/20，可作为独立观察。
- fixed step 200（λ=1）：Full 文本不变 18/20；Full 错误率 0.050（主 endpoint 不保证 Full 全对）。
- oracle best_strict：Full error 恒为 0 是选择器条件所致（FULL_SAFETY_IS_SELECTOR_CONDITIONED），不能单独证明攻击未伤 Full。

## 9. 主文应如何报告

1. 主结果报 fixed step 200 的 strict CSFR 与四格表。
2. min-margin 作为 post-hoc 回溯，标注 `POST_HOC_SELECTOR_AUDIT_NOT_PROSPECTIVE_EVIDENCE`。
3. 17/20 仅作 oracle 上界，标注 `GROUND_TRUTH_ORACLE_UPPER_BOUND`。

## 10. 不应声称什么

- 不应把 17/20 写成普通攻击成功率。
- 不应写「λ=0.25 显著优于 λ=1」除非未来预注册检验通过。
- 不应把 oracle 下的 Full error=0 等同于无选择器条件的 Full 安全。

## 与旧 final_report.md 的差异

- 旧报告以 oracle best_strict 作主表（λ=0.25 strict 17/20），与本次 `diagnostic_oracle_best_strict` 一致。
- 旧报告未单列 fixed step 200：重算主 endpoint 为 λ=0.25 15/20、λ=1 12/20。
- 旧报告写 holdout 两方法 Full 错误率均为 0，仅对 oracle 成立；λ=1 fixed step 200 为 1/20 Full ANLS<0.5。

## 11. 下一步预注册建议

- 正式实验主 endpoint 固定 final step（200）。
- 若部署用 min-margin，须事前预注册且选择时不得读 ANLS。
- 报告同时给出 oracle 上界作诊断，不与主 endpoint 混表。

