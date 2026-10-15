# 真实 PNG 有限差分诊断

本实验验证修复后的梯度方向能否预测真实 uint8 PNG 的响应。样本为 47450、60700，方向为同一条 FastV 核心 margin descent sign 方向，半径为 uint8 L∞=1、2。每个方向和半径重复生成两次，并独立回读 PNG。

结果：8 个 PNG 回读均通过 Full text exact、token exact 和 replay；每个重复点的 PNG hash、keep-set hash、FastV 文本和核心 soft margin 完全一致。说明真实像素路径在当前运行时是可重复的。

| sample | radius | 预测变化 | 自然核心 margin 变化 | 固定 keep-set 变化 | FastV 文本 |
|---|---:|---:|---:|---:|---|
| 47450 | 1 | -20.858 | -0.502 | -0.465 | 原答案 |
| 47450 | 2 | -41.716 | -0.776 | -0.763 | 原答案 |
| 60700 | 1 | -17.660 | -0.311 | -0.456 | `Ppbs` |
| 60700 | 2 | -35.320 | -0.522 | -0.587 | `Ppbs` |

反向方向的 margin 均上升，说明 sign 方向的方向性是正确的。预测幅度约比真实 PNG 响应大一个数量级；这是 sign 更新、低精度前向和离散像素量化共同造成的局部模型失真。自然 keep-set 与固定 keep-set 的变化方向一致，因此当前主要问题不是 keep-set 切换。

终态：`PNG_DIRECTIONAL_RESPONSE_PASS_NO_CONTENT_FAILURE`。

这个结果支持继续研究“带真实响应回读的步长校准”，但不支持宣称新的攻击方法已经成立：47450 没有改变答案，60700 只有大小写变化，按语义规则仍是答案保留。下一步若继续，只做少量带回溯的 trust-region 候选，每次真实 PNG 回读后接受；若 10–25 步仍不能产生 `CONTENT_FAILURE + Full exact`，停止这一方向。

证据：

- `pngfd2_gpu0/results.json`（47450）
- `pngfd2_gpu6/results.json`（60700）
