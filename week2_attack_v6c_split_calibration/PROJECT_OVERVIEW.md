# 多页文档视觉 Token 压缩攻击：当前思路与实验状态

> 状态快照：2026-10-09。  
> 当前主方法候选：**Variant B**。  
> 当前结论：开发证据支持“受控非证据辅图导致压缩路径单独失败”；真正未见的 `pristine60` 尚未开始。

## 1. 这项工作在研究什么？

视觉语言模型在处理多页文档时，常先把每页图像编码为视觉 token，再通过 FastV 等压缩器保留部分 token 供语言模型生成答案。完整路径（以下简称 **Full**）读取全部视觉 token；压缩路径（以下简称 **FastV**）只读取保留后的 token。

本工作的核心问题是：

> 若攻击者只能修改一张**非证据辅助页**，而证据页、问题、页面顺序都保持不变，能否使 FastV 压缩路径答错，同时让 Full 路径仍输出完全相同的答案？

这不是“让模型整体答错”的普通视觉攻击。目标是构造 **compression-only failure**：同一输入的 Full 路径正确且文本不变，而 FastV 路径错误。

## 2. 威胁模型与边界

每个 MP-DocVQA 样本含多张页面图像，记为 \(x=(x_1,\ldots,x_M)\)。

- 一张页面是问题所依赖的 evidence page；它始终保持干净。
- 攻击者仅能修改预先指定的一张 auxiliary page。
- 其他页面、问题文本、页面顺序、模型、压缩器和解码配置都固定。
- 扰动满足 \(\|\delta\|_\infty\le 8/255\)，最终以 uint8 PNG 进行评估。
- 攻击访问真实 LLaVA-OneVision、真实 FastV hook、问题以及 clean Full 路径生成的答案序列；因此这是一个**白盒、答案对齐**攻击。

本文使用“source-bounded”描述这一受限控制：攻击只作用于非证据辅助页，证据页受保护。它不自动等价于现实系统中的“可信来源/不可信来源”安全语义；若没有额外来源元数据，不作该延伸主张。

## 3. 主方法：Variant B

Variant B 是当前冻结的主方法候选。它由三部分组成。

### 3.1 压缩路径答案对齐攻击

令 \(y_t\) 为 clean Full 路径在第 \(t\) 个位置生成的 token。对真实 FastV 路径，定义该 token 的竞争 margin：

\[
m_t^{\mathrm{FastV}}=z_t(y_t)-\max_{j\ne y_t}z_t(j).
\]

攻击项 \(L_{\mathrm{comp}}\) 最小化这些 margin，使 FastV 路径不再偏好生成 clean Full 答案。答案 token 与一个 EOS token 都参与该项。

### 3.2 Full soft-margin anchor

Full 路径需要保持原回答。Variant B 不使用会在安全区出现零梯度的 hinge，而使用：

\[
L_{\mathrm{soft\text{-}full}}
=\frac{1}{|\mathcal T|}
\sum_{t\in\mathcal T}
\operatorname{softplus}\left(m_t^{\mathrm{clean}}-m_t^{\mathrm{adv}}\right).
\]

最终优化目标为：

\[
L_B=L_{\mathrm{comp}}
+\lambda_{\mathrm{soft}}L_{\mathrm{soft\text{-}full}},
\qquad
\lambda_{\mathrm{soft}}=2.049824826423478.
\]

该系数由冻结的梯度尺度校准得到，而非根据攻击成功率搜索。

### 3.3 Prospective feasible selector

每题以 Adam 优化 200 步，\(\epsilon=8/255\)、学习率 \(2/255\)，保存固定 checkpoint：

```text
10, 25, 50, 75, 100, 150, 200
```

最终不机械返回 step 200，而是在下列候选中选择 compressed margin 最低者：

- 扰动非零；
- uint8 \(L_\infty\le8/255\)；
- 输出有效；
- Full 解码文本与 clean Full 文本逐字相同。

并列时选更早 checkpoint。selector **不读取** ground truth、ANLS、FastV 解码文本或攻击是否成功。选图后，PNG 落盘重载并重新生成；重放失败则该题失败，不能换另一张图。

这三部分的职责不同：压缩损失让 FastV 偏离原答案，soft anchor 提供 Full 路径保护方向，selector 强制最终返回图满足 Full-exact 约束。

## 4. 实验设置

| 项目 | 当前设置 |
| --- | --- |
| 基础模型 | LLaVA-OneVision-Qwen2-7B |
| 压缩器 | FastV，\(r=0.25\) |
| 任务 | MP-DocVQA 多页文档问答 |
| 攻击权限 | 仅 auxiliary page；evidence/protected pages detach |
| 扰动 | uint8 \(L_\infty\le8/255\) |
| 优化 | Adam，200步，学习率 \(2/255\) |
| 主评估 | prospective feasible selector |
| 严格成功 | FastV ANLS < 0.5，且 Full 文本与 clean Full 完全一致、输出有效、重放一致、扰动非零且合规 |

Full 与 FastV 的真实解码、keep hash、K、PNG hash、uint8 \(L_\infty\) 和落盘重放均被保存。ground truth 与 ANLS 只用于评价，不用于 selector。

## 5. 已完成实验及结果

### 5.1 v5b：第一次扩展未见40题验证

`week2_attack_v5b_expanded_pristine40/` 曾在当时未参与攻击的同分布样本上验证旧方法。

| 指标 | 结果 |
| --- | ---: |
| fixed step 200 严格成功 | 33/40 |
| Full 文本逐字不变 | 39/40 |
| 随机噪声严格成功 | 3/40 |
| attack-only / noise-only | 30 / 0 |
| PNG 重放 | 120/120 |

该轮证明了 compression-only 现象存在，但样本 `60700` 的 Full 文本发生变化，因此未满足 Full-exact 40/40 门槛。此后这些40题被用于方法开发，不能再作为 Variant B 的独立泛化证据。

### 5.2 机制审计：内容与选择的2×2反事实

对 v5b fixed-step 的33个严格成功样本，运行 CC/CA/AC/AA：

- CC：clean content + clean keep-set；
- CA：clean content + adversarial keep-set；
- AC：adversarial content + clean keep-set；
- AA：adversarial content + adversarial keep-set。

| 机制类别 | 数量 |
| --- | ---: |
| CONTENT_SUFFICIENT | 21/33 |
| BOTH_INDEPENDENTLY_SUFFICIENT | 5/33 |
| INTERACTION_REQUIRED | 7/33 |
| SELECTION_SUFFICIENT | 0/33 |

因此内容具有充分性的样本为 \(21+5=26/33\)。这支持“受控辅图内容本身可导致压缩路径失败”的机制证据；但它不证明 keep-set 完全不变，也不支持 selection-stable 主张。

错误质量审计中，30/33 个严格成功属于核心答案改变或无关答案替换，而非主要依赖额外文本、截断或无效输出。

### 5.3 v6c：Variant B 开发40题确认

在已有40题上，Variant B 被冻结后重新运行。该40题此时是**开发确认集**，不是新的holdout。

| 指标 | Variant B |
| --- | ---: |
| 严格成功 | **38/40** |
| Wilson 95% CI | **[0.835, 0.986]** |
| Full 文本逐字不变 | **40/40** |
| 落盘重放一致 | **40/40** |
| 非零且uint8 \(L_\infty\) 合格 | **40/40** |
| Invalid output | **0/40** |
| 旧v5b single random noise | **3/40** |
| attack-only / noise-only | **35 / 0** |
| 精确 McNemar 双侧 \(p\) | **\(5.82\times10^{-11}\)** |

未达严格成功的两题是 `47450` 和 `52248`；两题的 Full 文本均保持 clean Full 原文，但 FastV ANLS 仍高于0.5。

Variant B 相比早期 Variant A 的点估计从35/40提升到38/40。不过这不是独立方法优越性证明：两种方法在同一开发数据上比较，配对差异为 B-only=4、A-only=1，精确 McNemar \(p=0.375\)。当前可以说 B 通过开发门槛，不能说 B 已统计显著优于 A。

### 5.4 B 与 C 的开发比较

| 版本 | strict / 12 | Full exact | 污染类错误 | 结果 |
| --- | ---: | ---: | ---: | --- |
| A：旧目标 + selector | 7/12 | 12/12 | 1 | 基线 |
| B：soft Full anchor | **11/12** | 12/12 | 1 | **选中** |
| C：B + 页级选择稳定项 | 10/12 | 12/12 | 2 | 未选中 |

C 的 median keep Jaccard 为0.831，低于 A 的0.836，也未提升硬Top-K选择稳定性。因此当前不使用 C，也不声称“selection-stable survivor poisoning”。

## 6. 当前可以与不可以声称什么

### 当前证据支持

1. 在当前设置下，存在 source-bounded compression-only attack：攻击仅改动非证据辅图，FastV 可被打错而 Full 输出保持逐字不变。
2. Variant B 在开发40题中达到38/40严格成功，并显著优于已保存的单随机噪声对照。
3. 已有2×2证据表明，许多成功具有 content-sufficient 特征。
4. Variant B 的最终攻击图在本次运行中可通过PNG落盘重放复现。

### 当前证据不支持

1. 38/40 不是未见样本泛化率；该40题已参与方法开发。
2. B 尚未被证明统计显著优于 A。
3. 不能声称 selection-stable survivor poisoning。
4. 不能声称该攻击适用于其他压缩率、其他压缩器、其他模型或所有视觉token压缩系统。
5. 不能把本工作的“受控辅助页/保护证据页”直接扩展成真实部署中的可信来源与不可信来源安全结论。

## 7. 当前未解决的工程与科学问题

攻击图可落盘重放，但完整优化过程对 FastV 的临界 Top-K 选择存在数值敏感性：同一方法的两次开发运行可能选择不同checkpoint或得到不同攻击图。pristine60 将把这一点作为预注册 repeatability 诊断，而非隐去。

此外，页级 soft selection mass 的稳定并不等同于硬 keep-set 稳定。C 的失败说明目前没有证据证明“稳定选择集时攻击仍保持同等强度”。

## 8. 下一步：pristine60 真正未见验证

下一轮计划为 `week2_attack_v7_pristine60/`，但截至本快照尚未启动。

核心原则：

1. 从同一 MP-DocVQA val 来源的历史未见多页题中一次性冻结60题；
2. 排除所有已攻击、已优化、已做机制审计或已查看攻击结果的ID；
3. 冻结 Variant B、候选池、页面hash、seed、selector、生成配置和通过门槛；
4. 以 prospective selector 为主指标，fixed step 200 仅作消融；
5. 保存 single random noise 与 selector-matched random trajectory 两种对照；
6. 对固定10题重复攻击，量化优化过程的重复性；
7. 对正式样本做冻结2×2机制审计。

建议的主要通过门槛：

- strict 至少38/60，且 Wilson下界大于0.5；
- Full exact、replay、nonzero、L∞均为60/60；
- INVALID_OUTPUT=0；
- 显著优于 selector-matched random trajectory。

若该轮通过，Variant B 可作为论文主方法的主要实证支撑；若不通过，应以 pristine60 的结果重新校准所有泛化主张。

## 9. 主要文件入口

| 内容 | 路径 |
| --- | --- |
| v5b首次扩展验证 | `week2_attack_v5b_expanded_pristine40/final_report.md` |
| v5b机制2×2 | `week2_attack_v5b_expanded_pristine40/mechanism_audit_v1/mechanism_report.md` |
| Variant B开发确认 | `week2_attack_v6c_split_calibration/final_dev_report.md` |
| Variant B冻结协议 | `week2_attack_v6c_split_calibration/FROZEN_PROTOCOL.json` |
| 系数冻结 | `week2_attack_v6c_split_calibration/FROZEN_COEFFICIENTS.json` |
| 当前开发决定 | `week2_attack_v6c_split_calibration/final_dev_decision.json` |

## 10. 给新读者的一句话

本项目发现并正在严格验证一种现象：在多页文档问答中，即使证据页完全不改，只对一张辅助页施加微小像素扰动，也可能让视觉token压缩路径答错，而完整模型仍保持原答案；Variant B 是当前用于验证这一现象的冻结主方法。
