# H0 更新轨迹法医诊断

状态：`RUNTIME_NONREPEATABLE`。分叉发生在纯净 Native-FDP 的 10 次更新内部，不是 v11a 的包装。

历史源码版本没有保存，记为 `HISTORICAL_SOURCE_PROVENANCE_INCOMPLETE`。当前命令不能称为历史命令。诊断仍按当前磁盘上的 `v7d.attack` 运行。

## 做了什么

1025 和 60700 各跑两次 10 步轨迹。两次都是新进程、新加载的模型，同一张 A100，GPU 0。直接调用历史 `v7d.attack(..., "native-FDP", steps=10)`。没有 import v11a，没有阶段 A，没有 generate / backward / forward hook / `generate_tokens` 包装。没有打开 deterministic algorithms，也没有改 TF32。记录到的当前状态是 `deterministic_algorithms=false`，`cudnn.deterministic=false`，`matmul_tf32=false`，`cudnn_tf32=true`。

## 结果

第 0 步在两次纯净运行和历史 Native-FDP 之间完全一致，包括 PNG、Full 文本、两个比例的 FastV 文本、keep-set 和 margin。

第 10 步，两次纯净运行彼此不一致，也都不等于历史记录。Full 文本仍然一致。PNG、keep-set 和 margin 都分叉。

| 样本 | 纯净 A，r=0.25 | 纯净 B，r=0.25 | 历史 r=0.25 |
|---|---|---|---|
| 1025 | Dr. C. Baur | Dr. Bausch | Dr. Bacharach |
| 60700 | PPBS | PPBS 14, 15, 1967 | PPBS 14, 1967 |

因此 v11a 在第 10 步对不上历史，不能归因于那些计数包装。去掉包装之后，同一条更新在当前运行时自己就不重复。60700 也是这样，不是 1025 单独的现象。

## 未运行

`C0_trace`、`C_all_v11a`、`C1` 到 `C5` 没有运行。纯净对照已经不可重复，这些 case 无法把包装从运行时噪声里分离出来。H1、P/O、噪声对照、未见样本、200 步六题都是 `NOT_RUN`。
