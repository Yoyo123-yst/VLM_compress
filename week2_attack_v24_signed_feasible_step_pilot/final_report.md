# Signed feasible-step pilot

终态：`NO_GO_SIGNED_FEASIBLE_STEP_PILOT`。

这是对 v23 差分方向失败原因的针对性修复，不是多步训练，也不是正式方法结果。实验只使用开发样本 47450、60700，在 FastV ratio=0.25 下测试半径 1 和 2。

## 修复内容

v23 的方向目标有两个问题：它使用无符号敏感度比，不能保证降低核心答案；并且连续方向随后被 sign-PGD 量化。v24 改为在逐 Full-token 线性约束下，直接最小化 FastV 核心 margin 的有符号局部二次目标，并保留 PNG 回读后的真实 Full exact/replay 检查。

第一次实现发现了一个执行符号错误：求解器返回的向量已经是下降步，保存 PNG 时又被取了负号。该次运行未计入结果；修正后重新启动两个独立进程，以下只统计修正后的 `final_gpu0` 和 `final_gpu6` 产物。

## 结果

| sample | radius | clean core mean | QP core mean | FastV text | Full exact/replay |
|---|---:|---:|---:|---|---|
| 47450 | 1 | 6.600 | 6.500 | `Dr. or Mrs. William J. Darby` | pass |
| 47450 | 2 | 6.600 | 6.550 | `Dr. or Mrs. William J. Darby` | pass |
| 60700 | 1 | 3.469 | 3.406 | `Ppbs` | pass |
| 60700 | 2 | 3.469 | 3.500 | `PPBS` | pass |

60700 的 `Ppbs` 仅改变大小写，按预注册语义规则属于答案保留，不是 `CONTENT_FAILURE`。47450 的核心实体完全没有改变。四个 PNG 均通过 Full text exact、token exact 和 replay；没有 OOM。

## 决策

这次修复证明有符号逐 token 约束可以在小半径内保持 Full，但没有产生新的语义攻击能力：`content_failure_rows=0/4`。因此不启动多步刷新、不扩展样本、不运行 unseen。

当前最合理的结论是：v23 的失败不只是 sign 量化造成的，局部可行步即使降低少量核心 margin，也不足以跨越 FastV 的离散生成边界。继续堆叠 QP、步长或权重没有证据支持。
