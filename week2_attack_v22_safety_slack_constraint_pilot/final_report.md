# 安全余量约束试点

状态：`NO_GO_SAFETY_SLACK_HINGE_ON_47450`

第二个样本：不允许。T3 没有运行，标记为 `NOT_RUN_UNVERIFIED_CONSTRAINED_BASELINE`。

## 研究问题

Native-FDP 的 `softplus(m_clean - m_adv)` 在 Full 输出仍安全时是否一直占用攻击容量。安全下限 hinge 只在 Full margin 接近预注册下限时才产生保护梯度。本轮只在 47450、FastV r=0.25 上检验这个假设，不是新方法结果。

## 冻结协议

预训练检查通过：clean keep-set 与 v21b 一致，CORE 的 uint8 L∞=8 一步图回读后 keep-set、processor tensor 和核心 margin hash 与 v21b 一致。τ_t = max(0.125, 0.5 m_clean,t)，在训练前写入 `preflight.json`。

三条轨迹共用 clean 图、CORE 位置 5–9、固定 clean FastV reference、sign-PGD、α=1/255、uint8 L∞≤8、λ=2.637、最多 25 次接受和 40 次尝试。前向像素来自 `processor.prepare`。selector 只比较合法 checkpoint 的 L_core，平局取更小 step。旧实验哈希在结束后没有变化。

现有代码里没有带投影顺序和回滚测试的 Orthogonal PGD 或 ALMA，所以没有临时写 T3。

## 轨迹

T0 在第 6 次尝试被 Full guard 拒绝后停止。拒绝时 Full 文本变成 `Dr. or Mrs. William J Darby`，token exact 失败，回滚后的 delta hash 与更新前一致。selector 留下的是仍合法的 step 5。

| step | L_core | CORE mean | Full min | Full exact | FastV |
|---:|---:|---:|---:|---|---|
| 0 | 6.114 | 6.600 | 0.625 | True | Dr. or Mrs. William J. Darby |
| 1 | 5.595 | 6.000 | 0.500 | True | Dr. or Mrs. William J. Darby |
| 2 | 4.928 | 5.150 | 0.375 | True | Dr. or Mrs. William J. Darby |
| 5 | 2.562 | 2.675 | 0.375 | True | Dr. or Mrs. William J Darby |

T1 接受了全部 25 步。step 0 的 attack 梯度与 FDP 梯度 cosine 约为 -0.28，FDP loss 约为 0.693。

| step | L_core | CORE mean | Full min | Full exact | FastV |
|---:|---:|---:|---:|---|---|
| 0 | 6.114 | 6.600 | 0.625 | True | Dr. or Mrs. William J. Darby |
| 1 | 6.186 | 6.750 | 1.000 | True | Dr. or Mrs. William J. Darby |
| 2 | 6.018 | 6.575 | 1.000 | True | Dr. or Mrs. William J. Darby |
| 5 | 5.357 | 6.138 | 1.375 | True | Dr. or Mrs. William J. Darby |
| 10 | 4.640 | 5.338 | 1.500 | True | Dr. or Mrs. William J. Darby |
| 15 | 2.109 | 2.513 | 1.500 | True | Dr. or Mrs. William J. J. Darby |
| 20 | 1.403 | 1.837 | 1.500 | True | Dr. or Mrs. William J. I. Darby |
| 25 | 2.973 | 3.325 | 1.375 | True | Dr. or Mrs. William J. Darby |

T2 的前 3 次更新里 hinge loss 为 0，梯度和 T0 同向；第 4 次尝试 hinge loss 变为 0.011，与 attack 梯度的 cosine 约为 -0.56。之后 40 次尝试里接受了 19 次，其余被 Full guard 拒绝。最后一次拒绝的 Full exact 已失败，核心 mean 降到 0.1625，但没有被接受。

| step | L_core | CORE mean | Full min | Full exact | FastV |
|---:|---:|---:|---:|---|---|
| 0 | 6.114 | 6.600 | 0.625 | True | Dr. or Mrs. William J. Darby |
| 1 | 5.595 | 6.000 | 0.500 | True | Dr. or Mrs. William J. Darby |
| 2 | 4.928 | 5.150 | 0.375 | True | Dr. or Mrs. William J. Darby |
| 5 | 3.223 | 3.350 | 0.250 | True | Dr. or Mrs. William J. Darby |
| 10 | 2.359 | 2.475 | 0.500 | True | Dr. or Mrs. William J. Darby |
| 15 | 2.176 | 2.325 | 0.250 | True | Dr. or Mrs. William J. Darby |

## Selector 与独立终点

| 轨迹 | selector step | r=0.25 文本 | 语义 | CORE mean | Full exact | L∞ | ANLS |
|---|---:|---|---|---:|---|---:|---:|
| T0 | 5 | Dr. or Mrs. William J Darby | ANSWER_PRESERVED | 2.675 | 通过 | 5 | 1.000 |
| T1 | 20 | Dr. or Mrs. William J. I. Darby | AMBIGUOUS_REQUIRES_REVIEW | 1.8375 | 通过 | 8 | 0.926 |
| T2 | 15 | Dr. or Mrs. William J. Darby | ANSWER_PRESERVED | 2.325 | 通过 | 8 | 1.000 |

r=0.15、0.35、0.50 只作诊断。T2 在 r=0.15 生成 `Daraby`，语义是 `AMBIGUOUS_REQUIRES_REVIEW`，不参与门槛。

## T1 与 T2

T2 的核心 mean 在 step 1 就降到 6.0，T1 在 step 1 升到 6.75，到 step 2 才降到 6.575。早期下降更快这一点成立，原因是当时 Full margin 还高于 τ，hinge 没有梯度。

selector 终点相反：T2 的核心 mean 是 2.325，T1 是 1.8375。T2 没有更低。T2 的 Full 最窄 margin 在 step 15 是 0.25，T1 在 step 20 是 1.5。T1 把 Full margin 抬高并继续接受更新；T2 在 margin 接近下限后开始被硬 guard 拒绝，没能把更低的核心 margin 留在合法 checkpoint 里。

两条轨迹的 r=0.25 自然输出都不是 `CONTENT_FAILURE`。T1 的 `I.` 插入按冻结规则是歧义，不算内容失败。

## 结论

预注册门槛要求 T2 的 selector 核心 margin 低于 T1，或在 Full exact 下出现内容失败。前者不成立，后者也不成立。因此停止这个 hinge，不扩大到第二个样本。早期更快的下降不能改写这个终点判断。
