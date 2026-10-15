# v24 梯度—真实前向一致性诊断（修复版）

## 目的

在继续设计新攻击前，检查当前 FastV 核心 margin 的梯度是否真的预测了执行后的方向。测试只使用开发样本 47450 和 60700，固定 clean keep-set，直接在连续像素空间评估 surrogate 前向；没有 PNG 选择变化、没有多步训练，也没有 unseen。

## 已通过的工程检查

- 数学求解器的下降方向、约束残差和像素边界测试通过。
- 原图与真实 uint8 ±1 像素扰动的 processor tensor 一致，预处理不是 RGB/归一化错位。
- 所有此前合法 PNG 的 Full text/token exact 和 replay 检查保持通过。

## 初版诊断的问题

初版把候选页重新送入了会先转换为 uint8 的 PIL 路径，同时梯度来自 straight-through 的 torch 缩放路径。这个前向和反向不是同一个函数，因此初版的“方向相反”不能作为模型梯度错误的证据。

修复版固定原始 `processor` 输入，只替换辅助页，并使用与 straight-through 反向分支相同的纯 torch 缩放路径。还增加了零步重复和正负方向中心有限差分。

## 修复版小步长结果

`predicted_change` 是当前 autograd 梯度对 sign 方向的一阶预测；`surrogate_soft_change` 是同一方向在修复后的连续 surrogate 前向中的实际变化；`central_fd_change` 是正负方向的中心有限差分。负值表示核心 margin 下降。

| sample | step (pixel) | predicted | actual | 解释 |
|---|---:|---:|---:|---|
| 47450 | 0.001 | -0.0209 | -0.0202 | 方向一致 |
| 47450 | 0.005 | -0.1043 | +0.0075 | 低精度下出现量化波动 |
| 60700 | 0.001 | -0.0177 | -0.0451 | 方向一致 |
| 60700 | 0.005 | -0.0883 | -0.0207 | 方向一致 |

两题的零步重复差异都是 0。更大的 0.01–0.125 像素步长仍出现非单调变化，一阶预测继续线性放大，说明方向大体可用，但幅度不能直接相信。

## 判定

终态：`DIAGNOSTIC_REPAIRED_DIRECTIONAL_RESPONSE_PLAUSIBLE`。

修复后，梯度方向在两个样本的极小步长上与中心有限差分基本一致；主要残差来自 bfloat16 前向、PIL/uint8 量化和 sign 步长造成的幅度失真。预处理和零步路径已通过，当前证据不支持“梯度方向完全错误”。

因此可以进入下一道很小的诊断门槛，但还不能启动多步攻击：用真实 PNG 的 ±1、±2 uint8 有限差分测量同一方向的实际核心 margin，并把额外生成成本计入。只有真实 PNG 响应在重复进程中稳定，才可以实现带回溯的 trust-region 更新。

## 后续唯一合理的工程方向

若要继续，先实现一个不依赖 straight-through 假梯度的响应估计：对预注册的少量 ±uint8 像素方向做真实 PNG 回读有限差分，并把额外 forward/generate 成本计入预算。只有有限差分在重复进程中稳定、且比 autograd 明显更能预测真实核心 margin，才值得构造 trust-region 或响应校准攻击；否则应停止这条攻击路线，将结果定位为严格的失败诊断和 Native-FDP 基线验证。

证据文件：

- `cont4_gpu0/results.json`（47450）
- `cont4_gpu6/results.json`（60700）
- `prep2_gpu0/results.json`、`prep2_gpu6/results.json`
- `calib2_gpu0/results.json`、`calib2_gpu6/results.json`
