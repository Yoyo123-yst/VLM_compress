# v19 Finite-Step Forensic Diagnostic

状态：`STAGE1_MIXED_NO_GO_STAGE2`

本目录是有限步长诊断，不是新攻击方法结果，也没有使用 unseen 样本。

## Stage 0：证据口径

- v18 的 Full text exact、FastV ANLS、seed 级结果分开保留；representation/layer replacement 标记为“内容+选择联合干预”，不再当作纯层定位。
- v13--v18 的结论只限于各自构造，不推广为整个机制不可能。
- 本轮所有状态来自已存在、可重放的 PNG；当前 vattack 运行时在实验前修复了 `huggingface-hub` 版本漂移（`transformers 4.57.0` 对应 `huggingface-hub 0.36.2`）。
- 60700 错误态的 PNG SHA 为 `9fa86ee802bd22064f6503ceed9b512b5fd0464022da66be3df73d3189f0a367`；7006 近答案态使用 B/checkpoints/step_010.png，SHA 与历史 trajectory 一致。

## Stage 1：设置

状态：60700 错误态、60700 恢复态、7006 近答案态；训练比例 `r=0.25`；每个状态比较自然选择与起点冻结 keep-set；步长为 `0.25/255, 0.5/255, 1/255, 2/255`；每条轨迹 4 次更新。

每次更新使用固定的起点 FastV 参考序列，目标为固定序列 soft margin 加现有 Native-FDP Full 保护。候选更新先投影到 `8/255`，再进行真实 Full 双重生成 exact/replay guard；不通过则不接受。记录了梯度 hash、L2/L∞、delta hash、PNG hash、keep-set、自然文本、ANLS 和 Full exact。

## 结果

| 状态 | 选择 | 结果 |
|---|---|---|
| 60700 错误态 `PPSB` | 自然 | 四档步长均回到 `PPBS/Ppbs`，无 strict |
| 60700 错误态 `PPSB` | 冻结 | `2/255` 末步得到 `Ppbs #6 (se)`，Full exact，FastV ANLS=0，strict 通过 |
| 60700 恢复态 `PPBS` | 自然 | `2/255` 末步得到 `Ppjs 14, 1967`，Full exact，ANLS=0，strict 通过 |
| 60700 恢复态 `PPBS` | 冻结 | `1/255` 末步得到 `PPBS 14, 1967`，Full exact，ANLS=0，strict 通过 |
| 7006 近答案态 `Tobacco Company` | 自然 | 四档步长均恢复 `RJ Reynolds Tobacco Company`，无 strict |
| 7006 近答案态 `Tobacco Company` | 冻结 | 四档步长均恢复 `RJ Reynolds Tobacco Company`，无 strict |

三条关键配置做了全新进程复跑，结果一致：60700 错误态冻结 `2/255` 仍为 `Ppbs #6 (se)`；60700 恢复态自然 `2/255` 仍为 `Ppjs 14, 1967`；7006 自然 `2/255` 仍在第 1 步恢复为 `RJ Reynolds Tobacco Company`。

## 判定

这不是“恢复路径覆盖机制已成立”的证据。它只证明：

1. 在 60700 这个特定状态，冻结离散选择可以改变有限步长轨迹，并重复保持一个合法错误端点；
2. 60700 的恢复态也可以在大步长下再次被推到错误答案；
3. 7006 的近答案状态对更新极其敏感，冻结选择没有阻止语义恢复。

因此 Stage 1 的结果是**跨状态混合**：有一个可重复的 selection effect witness，但没有在“近答案恢复”上形成稳定、优于自然选择的机制。按照预注册门槛，不启动 Stage 2 的候选恢复组因果干预，也不实现新的恢复路径攻击，更不扩展到六题或 unseen。

下一步若继续，只能做一个更小的因果核查：对 60700 的冻结 `2/255` 端点与 7006 的一步恢复端点，分别做等量随机 keep-set 对照和重新加载 replay；若 60700 的优势不显著高于随机，直接终止该路线。
