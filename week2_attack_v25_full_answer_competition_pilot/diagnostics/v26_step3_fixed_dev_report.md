# v26 Step 3 — 冻结目标开发验证

## 协议

六个固定顺序样本（47567、50779、51133、6194、62941、64506）在攻击前完成 clean-only 候选生成和目标冻结。每题运行 Native-FDP、complete-target、complete-front；预算分别为 20、13、10 个接受更新，保持 ratio=0.25、步长、扰动范围、Full exact guard 和 PNG replay 不变。39204 未纳入，因为没有合法的预注册完整错误目标。

## 固定 selector 结果

| 样本 | Native-FDP | complete-target | complete-front |
|---|---|---|---|
| 47567 | 未命中；终点 `2006` | 未命中；终点 `2600` | **第 7 步命中 `900`** |
| 50779 | 未命中；终点 `Dr. Donbaryx` | 未命中；终点 `Dr. Donbary` | **第 7 步命中 `Dr. Dorothy Darby`** |
| 51133 | 未命中；终点 `26` | 未命中；终点 `30` | 未命中；终点 `30` |
| 6194 | 未命中；终点 `01017-X00-XX-XX` | 未命中；终点 `01017-X0X-XX` | 未命中；终点 `0107-XX-XX` |
| 62941 | 未命中；终点 `October 13, 1979` | 未命中；终点 `October 13, 1979` | 未命中；终点 `October 13, 1979` |
| 64506 | 未命中；终点 `E` | 未命中；终点 `J` | 未命中；终点 `C` |

全部更新的 Full text/token exact、Full replay、FastV replay 和 uint8 L∞≤8 guard 均通过；没有 OOM。墙钟范围约 135–383 秒，complete-front 由于额外首分叉评分并不总是更快。

## 统计与解释

- 固定 selector 指定完整错误目标的命中率：Native-FDP **0/6**，complete-target **0/6**，complete-front **2/6**。
- front 的两个命中发生在 47567 和 50779，且都在第 7 个接受更新；这比同题的两个对照更早、更明确。
- 其余四题出现了不同程度的数字、日期、字符或答案片段变化，但没有命中预注册目标。它们不能直接替代固定 selector 的方法成功；应作为“候选目标未达成/表层变化”诊断状态。
- 目标覆盖率本身为 6/6，说明本轮主要失败点不是找不到目标，而是目标在给定预算和 Full 约束下的可达性、或候选目标与自然生成竞争路径不匹配。

## Step 3 判定

**开发集上出现了初步 discriminator witness，但不足以打开 unseen。** complete-front 在 2/6 新样本命中固定完整错误目标，而 Native-FDP 与 complete-target 为 0/6；这支持“首分叉约束可能帮助完整目标胜出”的局部信号。但样本数仍小，且 4/6 失败，不能写成稳定泛化或最终方法优势。

## Step 4：只修改一个因素

下一步不再改 loss、步长或 Full 权重。应做**目标可达性诊断**：对 6 个冻结目标记录 clean target score、首分叉 gap、实际自然输出的首分叉 token、目标序列后续位置分数和首次命中前的 Full 最小 margin，区分“首 token 没胜出”与“首 token 胜出但后续路径恢复”。这一步只读已有轨迹/PNG，优先不再烧 GPU。

只有当该诊断显示 front 的收益来自稳定的首分叉改善，才允许在同一冻结目标上做一次单因素预算扩展；否则将 complete-front 定位为有限的完整目标攻击基线改进，不继续堆模块。

Unseen 仍保持关闭。

