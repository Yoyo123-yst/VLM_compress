# 完整内容候选 trust-region 小试验

本轮修正了候选注册规则，只保留完整词或数字位置，排除标点、单字母、答案内部 tokenizer continuation 和原答案词片段。样本仍为 47450、60700，每题最多 5 步；每次更新必须经过真实 PNG 回读、Full text/token exact、replay，并使固定内容竞争 margin 下降。

结果：

| sample | 合法接受步数 | Full 合法最终文本 | 结果 |
|---|---:|---|---|
| 47450 | 5 | `Dr. or Mrs. William J. Darby` | 姓名没有变化 |
| 60700 | 2 | `PPBS` | 后续候选会产生 `Ppbs`，但 Full exact 失败并被回滚 |

60700 的候选在第 3 步以后确实继续降低内容竞争 margin，但候选图的 Full 输出不再 exact，因此不能接受。47450 的所有合法更新都保持原姓名。没有产生 `CONTENT_FAILURE + Full exact`。

终态：`CONTENT_COMPETITOR_NO_GO`。

这轮排除了“候选只是 tokenizer 词片段”这个工程问题，但完整内容候选仍没有解决 Full 保持与语义替换之间的冲突。当前不应扩展样本或打开 unseen；继续调步长和步数也没有新的证据支持。若要继续，必须从 Full 与 FastV 的内容可分离性重新做因果诊断，而不是继续堆叠同一类像素更新。

证据：

- `comp2_gpu0/results.json`（47450）
- `comp2_gpu6/results.json`（60700）
