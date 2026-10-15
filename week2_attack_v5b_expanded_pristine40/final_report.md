# v5b expanded pristine40

注册的完整通过标签 **没有达到**。固定 step 200 的严格 compression-only 成功是 **33/40**，Wilson 95% 区间为 [0.680, 0.913]，下界高于 0.5；随机噪声只有 **3/40**，exact McNemar p = 1.86e-9。未通过的唯一完整门槛是 Full 解码文本逐字不变：39/40，不是 40/40。

样本 `60700` 的 clean Full 输出是 `PPBS`，step 200 的 Full 输出是 `Ppds`。Full ANLS 仍为 0.75，FastV ANLS 为 0，所以它落在“FastV 错、Full 对”格里，但不算严格成功。

## 已观察事实

- 候选宇宙：val 唯一题 5187，多页题 3554，历史排除 280，未见多页题 3274。排除定义没有收窄，v5 的 `BLOCKED_INSUFFICIENT_PRISTINE_POOL` 没有改。
- Screen seed `2026100802` 抽出 200 题。Clean 合格 63。Final seed `2026100803` 抽出 40 题，与历史排除交集为 0。
- Frozen manifest canonical payload hash：`785761407e53fe9329866faa9d74063cdee31dc9531ce8527ee5a9a77fa88161`。
- Frozen manifest 文件字节 hash：`493eb58195496ededc6420c4a6a2500ba666c21cc8555db2311a3208541255b1`。
- Screen manifest canonical payload hash：`fb9bc10901cf8e2dcd8b9a6b769f1a391b86840d614d2ddee061a08615cc57fd`。
- Screen manifest 文件字节 hash：`5e97a229fbdf86fea45265daf08356e7552363b45424be53473fc1eed91b7ed7`。
- 历史排除表文件 hash：`cab0138e6de07433288e0f15fa428efabcf2362aadc3c28693889ef29b19918a`。
- 攻击配置未改：λ_full=0.25，200 步，ε=8/255，Adam，lr=2/255，v4 冻结尺度。主 endpoint 是 step 200，没有用 ANLS 选图。
- 四格（step 200）：FastV 错且 Full 对 34；两边都对 6；两边都错 0；Full 错且 FastV 对 0。INVALID_OUTPUT 0。
- 其中 34 格里有 1 题文本变化，因此严格成功是 33。未成功且文本未变的 6 题是 `7006`、`39204`、`47450`、`50779`、`62941`、`64506`，都是两边都对。
- final / min-margin / random-noise 的磁盘 PNG 重放为 120/120，失败 0。uint8 L∞ 40/40 通过。
- Min-margin 次指标严格成功 34/40。Oracle 上界 36/40，只作诊断。
- 配对：attack-only strict 30，noise-only strict 0。G 差均值 0.761，bootstrap 95% 区间 [0.634, 0.886]，不含 0。精确 sign-flip 因非零配对超过 20 对没有穷举；McNemar 与 bootstrap 方向一致，不构成互相矛盾。
- M 分布是描述，不是确认性分层：M=2 为 8/8，M=3 为 5/8，M=4 为 20/24。结果不是只由 M=2 撑起来的。

## 推断

在这一套 LLaVA-OneVision + FastV r=0.25、同一 MP-DocVQA val 分布上，冻结攻击在未见样本上把 FastV 打错、同时把 Full ANLS 留在 0.5 以上的比例，明显高于一次 ±8 随机噪声。这支持“效果来自优化方向”。它还没有满足预先写死的“Full 文本 40/40 逐字不变”，所以不能写成完整通过。

## 机制猜测

`PPBS` 变成 `Ppds` 说明 Full 路径的 token 分布被扰动了，但还没跨过 ANLS 0.5。这只是这一题的观察，不是压缩机制结论。

## 未测量

没有做 r=0.35、双辅图、第二模型、第二压缩器或 CC/CA/AC/AA。

## 运行

正式攻击 11:15 启动，四个 shard 都是退出码 0。

| Shard | GPU | 包装 PID | 结束 | 退出 |
|---|---|---|---|---|
| 0 | 0 | 4043441 | 14:43:44 | 0 |
| 1 | 1 | 4043552 | 15:00:45 | 0 |
| 2 | 4 | 4043754 | 14:22:19 | 0 |
| 3 | 6 | 4044016 | 14:38:19 | 0 |

噪声和重放随后在空闲的 0、1、4、6 上完成，退出码都是 0。50743 的 2 步和 25 步 smoke 只用于工程检查，没有进入这 40 题。

## 文件

- `week2_attack_v5b_expanded_pristine40/final_decision.json`
- `week2_attack_v5b_expanded_pristine40/analysis/statistics.json`
- `week2_attack_v5b_expanded_pristine40/results/fixed_step200.csv`
- `week2_attack_v5b_expanded_pristine40/results/min_margin.csv`
- `week2_attack_v5b_expanded_pristine40/results/oracle_diagnostic.csv`
- `week2_attack_v5b_expanded_pristine40/results/random_noise.csv`
- `week2_attack_v5b_expanded_pristine40/replay/shard_s*.json`
- `week2_attack_v5b_expanded_pristine40/EXECUTION_LOCK.json`
