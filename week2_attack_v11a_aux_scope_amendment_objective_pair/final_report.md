# v11a 辅助页范围修正与 H0 复现

阶段 A 通过。H0 六条轨迹已跑完，与历史 Native-FDP 从第 10 步起不一致。状态是 `BLOCKED_H0_REPLAY_DIVERGED`。H1 没有运行。

## 阶段 A

60700 和 7006 上，H0 与 H1 各做一步更新。优化器参数只有辅助页 `delta`。重建后的非辅助页逐元素差为 0，辅助页有变化。图像列表只替换 `aux_idx`。重载 PNG 的 uint8 L∞ 为 2。干净输入上固定参考与动态参考一致，teacher forcing 前缀与生成序列对齐。扰动后的 keep-set 来自当前输入的 FastV generate。

非辅助页的反事实梯度只记在 `context_gradient_observed`，没有当作通过或失败条件。

## H0

六题都跑了 200 步。第 0 步的 PNG SHA256 与历史 Native-FDP 一致。第 10 步起六题的 PNG 都不一致，代理损失也不一致。生成 token ID 没有写入历史曲线，记为未测。第 10 步的 Full 文本仍然一致。FastV 文本在部分题上已经不同：

| 样本 | 历史 r=0.25 | 本次 r=0.25 |
|---|---|---|
| 1025 | Dr. Bacharach | TAKEDA |
| 2008 | March 1968 | March 1956 |
| 47450 | Dr. or Mrs. William J. Darby | 相同 |
| 60700 | PPBS 14, 1967 | PPBS 14, 1567 |
| 7006 | RJ Reynolds Tobacco Company | 相同 |
| 39204 | FOOD TECHNOLOGY | Food technology |

47450 和 7006 的 r=0.25 文本相同，但 PNG、keep-set 和 margin 已经不同。历史动态选择器选出的步数也不再一致：1025 由 75 变为 200，47450 由 200 变为 50，60700 由 50 变为 200，7006 由 150 变为 75。2008 仍是 200，39204 仍是 100。

因此不能把后续结果拿来和这条已经漂移的 H0 比较。

六条并行墙钟约 47 分钟。单条耗时 1626–2795 秒。历史 raw 文件哈希仍然一致。

## 未运行

H1、共同封存选择、轨迹 oracle、P0/P1、O0/O1、噪声对照、0.15 与 0.50、未见样本均为 `NOT_RUN`。
