# 真实 PNG trust-region 小试验

有限差分门槛通过后，对 47450 和 60700 各运行最多 10 步。每步重新计算当前图像的梯度 sign 方向，依次尝试 uint8 半径 1、2；候选必须经过 PNG 回读、Full text/token exact 和 replay，并且核心 soft margin 严格下降才接受。

结果如下：

| sample | accepted steps | 最终 FastV 文本 | Full exact/replay | 语义结果 |
|---|---:|---|---|---|
| 60700 | 1 | `Ppbs` | pass | `PPBS` 的大小写变化，答案保留 |
| 47450 | 5 | `Dr. or Mrs. William J Darby` | pass | 删除句点，核心实体保留 |

60700 的第 1 步把核心 margin 从 2.6555 降到 2.3448；47450 的 5 步把核心 margin 从 6.1142 降到 3.0142。每个被接受的 PNG 都满足 Full exact/replay，但都没有改变姓名或实体内容。两个样本的流程都在出现表层文本变化后停止，避免把大小写或标点误报成攻击成功。

终态：`TRUST_REGION_DIRECTIONAL_ONLY_NO_CONTENT_FAILURE`。

这说明当前修复后的响应校准和真实 PNG 回读机制确实能稳定降低局部核心 margin，但它仍然只推动表层决策，不能跨过语义内容的生成边界。下一步不应扩大样本或进入 unseen；若要继续，必须改攻击目标，使候选竞争集中在非等价实体/数字 token，并继续使用同样的真实 PNG trust-region 验收。

证据：

- `trust_gpu0/results.json`（47450）
- `trust_gpu6/results.json`（60700）
