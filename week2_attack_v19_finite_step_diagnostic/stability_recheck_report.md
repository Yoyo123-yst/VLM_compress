# 60700 deterministic stability recheck

两条全新进程使用相同的 `60700_error` 起点、natural selection、content
objective、步长 `1.0/255`、4 次更新，并设置相同的确定性环境变量。

## Result

两条轨迹的每一步完全一致：

| step | FastV text | PNG/hash and keep-set |
|---:|---|---|
| 0 | `PPSB` | identical |
| 1 | `PPBS` | identical |
| 2 | `PPBS` | identical |
| 3 | `Ppjs` | identical |
| 4 | `Ppjs` | identical |

Full exact/replay 在两条轨迹的全部步骤均通过。也就是说，当前固定运行
环境下这条配置是可重复的，但它没有保持 `PPSB`；第 1 步恢复为正确
答案，最后进入 `Ppjs` 这一核心内容错误状态。

## Decision

`PPSB` 不能被描述为稳定 endpoint，只能描述为一个可复现的起点/中间
状态。当前 content objective 的稳定行为是“错误状态之间发生离散跳转”，
而不是持续保持同一个错误答案。因此下一步不应继续增加样本或进入 unseen；
应先验证一个状态保持机制（例如首次达到合法内容错误后冻结像素），并以
`CONTENT_FAILURE` 而非单一字符串 `PPSB` 作为 endpoint。

Artifacts: `results_stability_a.json`, `results_stability_b.json`。
