# 60700 错误状态保持消融

## 设置

- 同一 `60700_error` 合法起点（Full=`PPBS`，FastV=`PPSB`）；
- content objective、natural selection、步长 `1.0/255`、4 步；
- 对照：继续更新的 `results_stability_a.json`；
- 保持组：首次检测到 `CONTENT_FAILURE + Full exact` 后冻结像素。

## 结果

| step | 继续更新 | 冻结保持 |
|---:|---|---|
| 0 | PPSB | PPSB |
| 1 | PPBS | PPSB |
| 2 | PPBS | PPSB |
| 3 | Ppjs | PPSB |
| 4 | Ppjs | PPSB |

两组每一步 Full exact/replay 均通过；保持组 PNG hash 始终不变。

## 解释

冻结确实消除了从 `PPSB` 恢复到 `PPBS` 的现象，但它只是保存已经找到的
合法错误状态，并没有让攻击更容易找到该状态。因此这不是新的攻击能力
证据，而是一个正确的 checkpoint/state-retention 协议。

当前可支持的结论是：content objective 能找到内容错误，后续 FDP 更新会
破坏该错误状态；首次合法错误状态应被封存，不能继续无条件优化。下一步
若要形成方法贡献，必须在**从 clean 起点首次找到错误的成功率或成本**上证明
状态保持优于普通 Native-FDP，而不是只报告冻结后的 endpoint。

Artifacts: `results_hold_60700.json`, `results_stability_a.json`。
