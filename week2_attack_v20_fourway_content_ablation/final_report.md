# v20 四组内容目标消融

状态：`PILOT_SIGNAL_NOT_SUPPORTED`

这是 development ablation，分母是本轮 admission 后的 4 个干净正确样本。没有打开 unseen，也没有把 oracle checkpoint 当作方法结果。

## 样本

冻结开发顺序里的主集是 1025、2008、7006、39204、47450、60700，共 6 个。12 题协议里的其余样本本轮没有跑。

| 样本 | admission | 处理 |
|---|---|---|
| 1025 | `CLEAN_FASTV_CONTENT_WRONG` | Full 为 `Dr. C. Baur MD`，FastV 为 `Dr. C. Baur`。金答案姓氏是 Baum。核心 span 不可得。不进入主分母，也没有训练。 |
| 2008 | `CLEAN_FASTV_CONTENT_WRONG` | Full 为 `March 1968`，FastV 为 `March 1956`。只作为 clean-wrong 记录，不计入“正确内容被攻击成错误”。 |
| 7006、39204、47450、60700 | `ADMITTED_CLEAN_CORRECT` | 主分母，4/6。 |

语义 manifest 在攻击前冻结，SHA256 为 `ac60d7887066f655e21b914167b4999cad66273e333b988d0a9fec0eea78b704`。

## 四组实际优化的 token

训练比例只有 r=0.25。B/C/D 使用冻结的 clean FastV 序列。A 使用当前扰动图重新生成的序列。

- A：当前自然 FastV 的全部 token，包含 EOS。60700 上这个序列会离开 clean 参考，例如中途变成 `Ppbs` 的 token。
- B：冻结 clean 序列的全部 token，包含 `<|im_end|>`。
- C：同一冻结序列去掉 EOS。四个主样本里 C 都不含 `<|im_end|>`。
- D：预注册核心 span。匹配失败时状态是 `CORE_SPAN_UNAVAILABLE`，不会改用 C 的位置。

D 与 C 的位置集合：

| 样本 | 关系 | D 的 token |
|---|---|---|
| 7006 | 与 C 相同 | `R J Reynolds Tobacco Company` |
| 39204 | 与 C 相同 | `FOOD TECHNOLOGY` |
| 60700 | 与 C 相同 | `PP BS` |
| 47450 | D 是真子集 | `William J. Darby`；C 还包含 `Dr. or Mrs.` |

7006、39204、60700 上 C 与 D 的梯度 hash 和 checkpoint PNG hash 完全一致。这三题不能用来证明核心位置选择有独立贡献。只有 47450 能做这个对照，而且 D 的梯度与 PNG 都不同于 C。

## 主结果

固定 selector：合法 PNG 中该组 attack surrogate 最低，平局取更早 checkpoint。它不读取 ANLS、语义标签或是否攻击成功。

主成功定义：独立进程重读 PNG 后，r=0.25 为 `CONTENT_FAILURE`，同时 Full text/token exact、Full replay、输出有效、uint8 L∞≤8。

| 方法 | 固定 selector 成功 | 选出的文本 | oracle 任一合法内容失败 |
|---|---:|---|---:|
| A dynamic | 1/4 | 47450 `Nashville, Tenn.` | 1/4 |
| B fixed-all | 1/4 | 47450 `THE NUTRITION FOUNDATION, INC.` | 1/4 |
| C fixed-noEOS | 1/4 | 47450 `THE NUTRITION FOUNDATION, INC.` | 2/4 |
| D core | 0/4 | 47450 停在 `Dr. or Mrs. William J. I. Darby`，标签为人工审核 | 1/4 |

7006、39204、60700 的固定 selector 都没有内容失败。39204 只有大小写差异，按规则保留。60700 的 A 是 `PPBS #6 (14, 15, 16)`，规则记为带后缀的保留；B 的 `PEPS`、C/D 的 `HEALTH OBJECTIVES` 不在预注册失败表里，记为人工审核。这些输出的 ANLS 可以是 0，但没有计作内容攻击成功。

oracle 与固定 selector 的差距：A 和 B 为 0；C 和 D 各多出 7006 的 checkpoint 21，差距是 1/4。该 checkpoint 的文本是 `RJ Reynolds Tobacco Co.`。冻结分类器把它标成实体替换。它仍是同一机构的缩写，固定 selector 没有选它；继续更新后，C/D 的 selector 回到了完整公司名。这是成功后丢失，不是方法主结果。

匹配随机噪声：每个 admitted 样本 3 张 L∞=8 且通过 Full exact/replay 的噪声 PNG。12/12 在 r=0.25 上保持原答案，内容失败率 0。A/B/C 相对噪声的主指标净差是 1/4，D 是 0。噪声本身没有制造内容失败。1025 没有进入分母，因此没有把历史高噪声样本算进方法收益。

## 对四个问题的直接回答

1. B 没有高于 A。两者都是 1/4，而且成功发生在同一个 47450 上。固定参考没有表现出对目标漂移的稳定优势。
2. C 没有高于 B。两者都是 1/4。排除 EOS 没有带来额外的固定 selector 成功。
3. D 没有高于 C。唯一 token 集合更小的 47450 上，C 选出了内容失败，D 只插入了 `I.`，规则要求人工审核。核心位置选择没有独立贡献证据。
4. 本轮没有从错误起点把“保持错误”计成 clean-start 成功。2008 的 FastV 在攻击前已经是 `March 1956`，未训练。
5. A/B/C 的点估计高于这 12 张噪声图，但只来自一个样本，而且三个预注册对照都不成立。

`PPSB` 没有出现在这次 clean-start 正式轨迹的独立评测里。它不是本轮可重复的 clean-start 结果。

## Full、预算与 keep-set

全部封存 checkpoint 的独立评测都通过 Full text exact、token exact 和 replay。uint8 L∞ 均不超过 8。60700 的 C/D 在 40 次尝试里只接受了 15 次，另外 25 次因 Full guard 失败被回滚；回滚后的接受步没有 Full 失败。

25 次接受的轨迹大约是 167 次 forward、50 次 backward、103 次 generate，墙钟约 5–8 分钟。60700 的 C/D 因为拒绝更多，forward/backward/generate 升到 230/80/140。比较的是同一次接受预算，不是忽略拒绝成本后的步数。

选出 checkpoint 的 keep-set 与 clean keep-set 的 Jaccard 约在 0.81–0.87。选择集合有变化，但没有变成另一套完全无关的下标。

其他比例只作迁移记录。47450 A 的内容失败没有迁移到 0.15、0.35 或 0.50。这些比例没有参与选 checkpoint。

## 是否扩大

不值得为了论文结论扩到 12 个 development 样本，也不允许打开 unseen。主对照没有成立；四题里有三题的 D 与 C 是同一个 token 集合。再增加样本只会在同样的目标定义下累积 pilot 计数，不能把当前结果写成泛化或核心内容机制。

若以后还要检验核心位置，前提是样本的预注册 span 必须严格小于 noEOS 集合，并且 admission 时 clean FastV 已经正确。1025 和 2008 当前都不满足。

## 预注册问题

1. A 优化当前自然 FastV 全序列；B 优化冻结 clean 全序列并保留 EOS；C 优化同一冻结序列但去掉 EOS；D 只优化预注册核心 span。
2. 当前 core-content 只在 47450 上不同于 fixed-noEOS。7006、39204、60700 上两者的位置、梯度和 PNG 相同。
3. clean-start 主指标上 A、B、C 都是 1/4，D 是 0/4。没有单独更好的一组。
4. 固定 selector 成功率：A 1/4，B 1/4，C 1/4，D 0/4。
5. oracle 与固定 selector 的差距：A 为 0，B 为 0，C 为 1/4，D 为 1/4。多出来的是 7006 checkpoint 21 的 `RJ Reynolds Tobacco Co.`，selector 没有选它。
6. 封存 checkpoint 的独立评测全部通过 Full text exact、token exact 和 replay。被 Full guard 拒绝的更新已回滚。
7. 12 张匹配噪声图的内容失败率是 0。A/B/C 比噪声高 1/4，D 与噪声同为 0。这只是一个样本上的点估计。
8. `PPSB` 没有出现在本轮 clean-start 正式评测中，不能写成可重复的 clean-start 攻击。
9. 固定参考、去掉 EOS、核心位置选择都没有得到“高于对照”的证据。状态保持也没有被计成 clean-start 成功。
10. 不值得为了更强结论扩展到 12 个 development 样本。
11. 不允许打开 unseen。
