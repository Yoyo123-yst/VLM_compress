# Branch reachability route closeout

状态：`NO_GO_BRANCH_SELECTION_AND_CROSS_BUDGET`

这次 closeout 只汇总 60700 的已注册因果消融，不是六题结果，也不是
unseen 结果。旧的 `final_report.md` 保留为构造实验历史记录；本文件记录
其后完成的 state-retention 和 cross-budget gate。

## 已证实的正结果

- 固定 AWARE 分支为 `BS -> SB`，训练比例 `r=0.25`，Full 输出为 `PPBS`。
- 在第 11 个 state step，固定 pair gap 首次达到 `-0.25 <= 0`，且两次真实
  Full generate 均保持 text/token/replay exact。
- 从该合法状态冻结 `delta` 和 Adam 状态；到 state step 25 不再更新像素。
- step 11、15、20、25 的冻结 PNG 哈希相同：
  `9fa86ee802bd22064f6503ceed9b512b5fd0464022da66be3df73d3189f0a367`。
- 训练进程与两个全新进程重放均得到 Full `PPBS`、FastV `PPSB`、ANLS `0`；
  Full exact/replay、FastV replay 和 `L_inf=8` 均通过。

这建立了一个有限的因果解释：原 AWARE 在达到分支目标后继续受到 Full-FDP
更新，因此可以从错误状态漂回正确答案；冻结状态能消除本样本的 step-25
恢复。它没有证明 AWARE 的选择准则优于其他选择器。

## 未证实且按停止规则关闭的部分

原始构造轨迹中，60700 的首次严格成功 checkpoint 是：

| variant | first strict checkpoint |
|---|---:|
| SEQ | 10 |
| NAIVE (`PP -> Pages`) | 10 |
| AWARE (`BS -> SB`) | 15 |

AWARE 虽然局部 `Full capacity / FastV radius` 比值更高，却比 SEQ/NAIVE 晚
成功；因此没有 selector advantage 证据。冻结只修复保持性，不改变这个比较。

随后只对同一张冻结 PNG 尝试 `r=0.35`，同时保持 `r=0.25` 已完成分支和
Full exact。15 次更新全部被拒绝；第一次候选更新就把 r=0.25 pair gap 从
`-0.125` 推到 `2.5`，触发保持守卫。PNG 未改变，独立重放仍为 r=0.25
`PPSB`、r=0.35 `PPBS`，没有新增预算 endpoint。

## 结论和边界

当前“Full-aware 分支选择 + 冻结保持 + 跨预算状态机”组合应停止扩大：

- 分支定向攻击在该设置可行，但只是可行性结果；
- Full-aware 局部选择没有优于 SEQ/NAIVE 的证据；
- 冻结是恢复原因的因果消融，不是新的选择机制收益；
- 跨预算调度没有产生第二个自然错误答案；
- 没有扩大到六题或 unseen，也不应把这些结果写成通用方法结论。

对应产物：

- state retention: `runs/state_retention_v1/60700/AWARE_FREEZE/trajectory.json`
- fresh replay: `runs/state_retention_v1/60700/AWARE_FREEZE/step_011.fresh_replay.json`
  和 `step_025.fresh_replay.json`
- cross-budget: `../week2_attack_v14_cross_budget_state_gate/final_report.md`
  和 `final_decision.json`
