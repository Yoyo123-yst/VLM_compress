# Full 安全余量扫描

状态：`FULL_SAFETY_SLACK_CONFIRMED`

下一步：可以之后提议一个 10–25 步的受约束比较。本轮没有启动 Native-FDP、safety-hinge、OPGD 或 ALMA。

样本只有 47450。方向是 v21a 的 attack-only 符号，梯度哈希与 v21a 一致：CORE `ee415200ee139c60…`，ALL `87ea0860cbc1c88c…`。每个半径都从 delta=0 单独投影。实际 uint8 L∞ 都等于目标值。12 张 PNG 的独立复评与扫描记录一致。v20/v21/v21a 的冻结哈希没有变化。

## CORE

prefix-safe radius 是 8。最大合法测试半径是 8。非单调是 False。

| 半径 | 实际 L∞ | Full exact | Full hard min | 最弱 token | FastV CORE mean | FastV 文本 |
|---:|---:|---|---:|---|---:|---|
| 0 | 0 | True | 0.6250 | ` or` | 6.6000 | Dr. or Mrs. William J. Darby |
| 1 | 1 | True | 0.5000 | ` or` | 6.2000 | Dr. or Mrs. William J. Darby |
| 2 | 2 | True | 0.5000 | ` or` | 5.8250 | Dr. or Mrs. William J. Darby |
| 4 | 4 | True | 0.3750 | ` or` | 5.5000 | Dr. or Mrs. William J. Darby |
| 6 | 6 | True | 0.3750 | ` or` | 5.2500 | Dr. or Mrs. William J. Darby |
| 8 | 8 | True | 0.3750 | ` or` | 5.2250 | Dr. or Mrs. William J. Darby |

## ALL

prefix-safe radius 是 8。最大合法测试半径是 8。非单调是 False。

| 半径 | 实际 L∞ | Full exact | Full hard min | 最弱 token | FastV CORE mean | FastV 文本 |
|---:|---:|---|---:|---|---:|---|
| 0 | 0 | True | 0.6250 | ` or` | 6.6000 | Dr. or Mrs. William J. Darby |
| 1 | 1 | True | 0.5000 | ` or` | 6.2250 | Dr. or Mrs. William J. Darby |
| 2 | 2 | True | 0.3750 | ` or` | 5.9750 | Dr. or Mrs. William J. Darby |
| 4 | 4 | True | 0.2500 | ` or` | 5.5750 | Dr. or Mrs. William J. Darby |
| 6 | 6 | True | 0.2500 | ` or` | 5.4750 | Dr. or Mrs. William J. Darby |
| 8 | 8 | True | 0.2500 | ` or` | 5.3250 | Dr. or Mrs. William J. Darby |

## 问题

1. CORE 和 ALL 的 prefix-safe radius 都是 8。1、2、4、6、8 全部通过 Full text/token exact、replay 和独立复评。
2. 扫描范围内没有 Full exact 失败。最接近边界的 token 一直是 Full 序列里的 ` or`。干净时它的 hard margin 是 0.625；CORE 半径 8 降到 0.375，ALL 半径 8 降到 0.25。
3. 有。从半径 1 开始，Full exact 仍成立，而这个 token 的 margin 已经低于 1，最低到 0.25。它没有跨到 0，也没有改写 Full 文本。
4. FastV 核心姓名的 hard-margin 均值随半径下降。CORE 从 6.6 降到 5.225，ALL 从 6.6 降到 5.325。
5. FastV 自然输出没有变化，两张方向的全部半径都仍是 `Dr. or Mrs. William J. Darby`。诊断语义是 `ANSWER_PRESERVED`，没有参与半径选择。
6. 在这条固定的一步符号方向上，uint8 L∞ 到 8 仍保持 Full exact，同时 FastV 核心 margin 下降。这说明 always-active 的 Native-FDP 可能在 Full exact 仍然允许的区域内继续把更新推回去。这只是一个样本上的一步扫描，不是多步攻击已经成立。
7. 终态允许之后提议小型受约束优化比较。本轮没有运行该比较。
