# Response calibration diagnosis

终态：`BLOCKED_DIFFERENTIABLE_RESPONSE_MISMATCH`。

无模型数学测试已通过后，在 47450 和 60700 上做了少量一步响应诊断。每个样本测试半径 1、2，记录了梯度预测、PNG 回读后的自然 keep-set、冻结 clean keep-set、核心 token margin、自然文本和 Full exact/replay。

## 关键结果

| sample | direction/radius | predicted core change | natural core mean | fixed keep core mean | FastV text | Full exact |
|---|---|---:|---:|---:|---|---|
| 47450 | RAW/1 | -20.858 | 6.000 | 6.025 | 原答案 | pass |
| 47450 | QP/1 | -0.534 | 6.500 | 6.625 | 原答案 | pass |
| 47450 | RAW/2 | -41.716 | 5.650 | 5.650 | 原答案 | pass |
| 47450 | QP/2 | -1.068 | 6.550 | 6.625 | 原答案 | pass |
| 60700 | RAW/1 | -17.660 | 3.188 | 3.094 | `Ppbs` | pass |
| 60700 | RAW/2 | -35.320 | 2.969 | 2.969 | `Ppbs` | pass |

60700 的 `Ppbs` 只是大小写变化，不是内容失败。60700 的 QP 在逐 Full-token 约束下没有合法候选，因此没有伪造 QP 结果。

## 诊断

固定 keep-set 并没有让预测变得准确。以 47450 为例，RAW 半径 1 的预测下降约 20.86，但实际固定 keep-set 只下降约 0.58；半径 2 的预测约 41.72，实际约 0.95。自然 keep-set 与固定 keep-set的实际变化接近。

因此当前证据支持“可微梯度路径与真实 PNG/processor 响应不一致”，暂时不能把问题归因于 FastV 动态选择，也不能据此提出路径校准攻击。

根因线索在 v24 复用的 straight-through 路径：前向值固定为已有 processor tensor，只有反向信号经过 `normalized_tiles`。这保证了前向输入不变，但不保证该反向导数能预测真实 PNG 回读后的有限差分。

## 决策

第三阶段方法比较没有启动。下一步必须先做导数一致性修复：在小尺寸和真实页面上比较 surrogate processor 的有限差分与 PNG/processor 有限差分；只有误差达到可接受范围，才重新评估 Native-FDP、QP 和校准方向。当前没有依据继续扩大样本、训练多步或运行 unseen。
