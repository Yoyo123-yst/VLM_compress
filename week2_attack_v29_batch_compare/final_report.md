# v29 batch comparison — final report

## Status

`DEVELOPMENT_SEMANTIC_SIGNAL_REQUIRES_FROZEN_VALIDATION`

本轮已经完成：120 个冻结候选的 clean screen、30 个固定顺序的 development 样本选择、30 个 clean-only target review、15 个合法完整错误目标的攻击，以及 45/45 张封存 PNG 的独立进程复评。未打开 20 个 unseen 样本。

## Cohort and target coverage

- 冻结队列：120；clean eligible：56；按预注册顺序选取：30。
- 其中有合法、完整、语义明确的错误目标：15/30。
- 其余 15 个为 `NO_VALID_PREREGISTERED_COMPLETE_WRONG_TARGET`，没有为了凑成功率换题。
- 三种方法只在这 15 个目标有效样本上比较；目标不可用的 15 个没有伪造攻击结果，单独作为 coverage 限制报告。

## Independent endpoint checks

45/45 张 PNG 均在独立进程中重新加载模型评估。三种方法均为：

- Full text exact：15/15；
- Full token exact：15/15；
- 同一评估进程重复回放一致：15/15；
- PNG 扰动满足 uint8 L∞ 预算（complete-front 有一张量化后实际为 6，其余为 8）。

因此，本轮没有出现 Full 破坏、PNG replay 分叉或评估不确定性。

## Fixed-target comparison

| 方法 | 目标精确命中 | FastV ANLS<0.5 | Full exact |
|---|---:|---:|---:|
| Native-FDP | 不适用（非定向） | 7/15 | 15/15 |
| complete-target | 7/15 | 6/15 | 15/15 |
| complete-front | 6/15 | 6/15 | 15/15 |

目标命中只说明生成了预先指定的错误答案，不能替代内容错误指标。ANLS 也不能替代语义审核：例如年份从 1985 变为 1988、姓名从 Fred 变为 John，仍有较高 ANLS，但已经是明确答错。

## Question-specific semantic audit

对最终自然生成文本按问题和数据集答案重审后，`CONTENT_FAILURE` 分别为：Native-FDP 6/15、complete-target 8/15、complete-front 9/15；对应的逐题理由见 `semantic_endpoint_audit.md`。无效或含混输出与内容错误分开报告。

## Interpretation

ANLS 统计不支持方法优势，但它低估了数值、年份和实体替换造成的实际答错。采用预先写明的逐题语义规则后，complete-front 在这 15 个开发样本上比 Native-FDP 多 3 个明确内容错误，比 complete-target 多 1 个。这个信号来自开发集，不能当作泛化结论。目标 coverage 只有 50%，仍是方法的主要限制。

因此本轮结论是：

1. complete-front 具有值得验证的开发集语义信号；
2. complete-target 仍是必要的定向攻击基线；
3. 不应打开 unseen；
4. 下一步应冻结当前 complete-front 协议，在新的、未参与本轮方法选择的开发样本上完成一次小规模验证，并继续单独报告目标覆盖率。

## Provenance

训练结果在 `attacks/`，独立复评结果在 `independent_eval/`。本报告只使用最后一个固定 selector 选出的 PNG，不使用 oracle checkpoint 或“轨迹中曾经成功”的状态。
