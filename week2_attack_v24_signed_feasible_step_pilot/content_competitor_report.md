# 核心内容竞争目标小试验

本实验在已有真实 PNG trust-region 流程上，只替换 surrogate 目标：对注册的核心位置选择 clean logits 中的非等价竞争 token，降低固定的 `z(reference)-z(competitor)`。每题最多 5 步，每一步都要求 PNG 回读、Full text/token exact、replay 通过，且竞争 margin 真实下降。

结果：

| sample | 接受步数 | 最终文本 | 语义变化 |
|---|---:|---|---|
| 60700 | 1 | `Ppbs` | 大小写变化，`PPBS` 保留 |
| 47450 | 4 | `Dr. or Mrs. William J Darby` | 标点变化，姓名保留 |

所有接受更新都通过 Full exact/replay。竞争 margin 确实下降，但没有出现实体替换或数字错误。候选记录也暴露出当前自动筛选规则的问题：部分候选是 `Dar`、`D`、`aby` 这样的 tokenizer 片段，不能直接当作有意义的非等价实体候选。

终态：`CONTENT_COMPETITOR_SURFACE_ONLY`。

这轮不能作为新攻击方法的成功证据。下一步若继续，只修正候选注册：排除标点、大小写变化、原答案内部词片段和 tokenizer continuation，只保留完整非等价词或数字候选；然后仍用同一套真实 PNG trust-region 验收。若合法候选仍只能产生表层变化，应停止这条路线。

证据：

- `comp_gpu0/results.json`（47450）
- `comp_gpu6/results.json`（60700）
