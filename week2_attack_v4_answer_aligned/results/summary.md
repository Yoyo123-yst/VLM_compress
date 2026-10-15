# v4 answer-aligned 6 题

决定：**WEAK_SIGNAL**

健康检查：PASS。这 6 题是开发诊断集，不能当成泛化结果。

## 逐题（best checkpoint，不是最后一步）

| sample | 组 | best step | 标签 | replay | Full ANLS | FastV ANLS | Full 答案 | FastV 答案 | margin 起 → best | 证据页保留率 | 辅图份额 |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| 50743 | 当前失败、oracle 可失败 | 200 | SUCCESS_COMP_ONLY | 1 | 1.000 | 0.000 | 10-K | I | 5.67 → −12.00 | 0.186 | 0.281 |
| 4912 | 当前失败、oracle 可失败 | 200 | SURROGATE_NONFLIP | 1 | 1.000 | 1.000 | Joe Mitchell | Joe Mitchell | 3.40 → 2.38 | 0.309 | 0.181 |
| 56776 | 当前失败、oracle 可失败 | 150 | SURROGATE_NONFLIP | 1 | 1.000 | 0.667 | Mar | May | 3.16 → −1.56 | 0.213 | 0.409 |
| 49242 | 历史成功 | 150 | SURROGATE_NONFLIP | 1 | 0.905 | 0.808 | GENEAL FOODS FUND, INC. | GENERAL R & D FOODS FUND, INC. | 3.89 → 1.80 | 0.286 | 0.321 |
| 33923 | 历史成功 | 200 | SUCCESS_COMP_ONLY | 1 | 1.000 | 0.000 | REGIONAL BUREAU ACTIVITIES IN NCF | Program Monograph | 5.01 → −12.32 | 0.139 | 0.327 |
| 5169 | oracle 也打不掉 | 200 | SUCCESS_COMP_ONLY | 1 | 1.000 | 0.000 | Mg | M | 5.48 → −13.33 | 0.308 | 0.286 |

标签计数：SUCCESS_COMP_ONLY = 3，SURROGATE_NONFLIP = 3，FULL_LEAK = 0，INVALID_OUTPUT = 0。

三个“当前失败但 oracle 可失败”的样本里，新增严格成功 **1/3**（只有 50743）。56776 从 Mar 变成 May，ANLS 仍是 0.667。4912 答案没变。

两个历史成功里保住 **1/2**（33923）。49242 的 FastV ANLS 最低到 0.808，没有再次落到 0.5 以下。

6/6 的 uint8 重载 L∞ = 8/255，6/6 replay 与 best checkpoint 的原文和 keep hash 一致，6/6 不压缩 ANLS ≥ 0.5。

## 保留集和答案是否一起变

没有。六题的证据页保留率和辅图份额相对 step 0 的变化都在 0.01 以内。相邻检查点的 keep-set Jaccard 大约 0.88–0.94，集合在换个别 token，但页份额不变。

答案变化因此不是证据页被挤出保留集。压缩模型读到的仍是几乎同一批 token，其中受控页上活下来的 token 被像素改过。不压缩模型看得到全部 token，答案保持原样，Full KL 在 0.0003–0.0013。

## 5169

Oracle 把证据 token 全部转走之后，这题的 ANLS 仍是 1.0。这次攻击的证据页保留率停在 0.308，和干净时的 0.307 一样，FastV 答案却从 Mg 变成 M，ANLS 变成 0。

这不是证据页删除。保留下来的表示被改了，压缩路径只读这些存活 token，所以一个字母的差别就跨过 ANLS 0.5。不压缩路径仍输出 Mg。属于 survivor poisoning。

50743 和 33923 的成功同样发生在页份额几乎不动的时候。50743 的 FastV 答案变成 I，33923 变成 Program Monograph。

## 健康检查里记下的精度

K 都是 3699，和官方 round 一致。真实 FastV hook 和 generate 的保留集一致，首 token argmax 一致，重放一致，历史解码一致。受控页梯度有限且非零；保护页特征是 detach 的。

另一条手写打分和 hook 的分数最大绝对差约 1.3e-5，但门槛附近约 2500 个分数挤在 1e-5 以内，top-k 会换掉几个 token。攻击没有用那条手写打分，每步的保留集来自真实 FastV hook。

## 运行

- CPU 测试：`tests/test_v4_cpu.py`，通过。
- 健康检查：GPU 4，`run_health.py`，退出 PASS。
- 尺度：50743 上 δ=0 的压缩梯度 L2 是 20.85。KL 在 δ=0 的梯度 L2 只有 1.7e-5，当作最小值处的数值灰尘，改用固定的 +1/255 平移，L2 是 0.0282。两项相除后 λ=1，六题共用，没有逐题调。
- 2 步和 25 步冒烟在 GPU 4 上完成，没有 OOM 或 hash 错误。25 步时 50743 已出现一次可重放的 FastV 答错。
- 正式 200 步：GPU 4 跑 50743、56776、33923；GPU 6 跑 4912、49242、5169。命令是 `run_smoke.py --steps 200 --tag official --shard {0,1} --nshards 2`。进程 3449261、3449262。日志 `logs/official_s0.log`、`logs/official_s1.log`。单题约 25 分钟，墙钟约 74 分钟。两个进程都正常结束，没有 Traceback。

不扩到 20 题。下一档才是一次受控的优化修改；这次没有加 soft Top-K，也没有上双辅图。
