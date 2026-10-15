# Complete-front vs complete-target：冻结开发集验证

## 结论

本轮结论为 **`COMPLETE_FRONT_SINGLE_SAMPLE_EARLY_COST_SIGNAL_NOT_CONFIRMED`**。

complete-front 在 61240 上比 complete-target 早 3 步命中同一个完整错误答案，少用了 3 次反向计算；但只有一个 clean 合格样本出现目标命中。两者在最终图片上的结果相同，4742 上两者都没有命中目标。按预先写下的规则，这不足以证明 complete-front 稳定优于 complete-target。当前应保留 complete-target 作为较简单的定向攻击对照；complete-front 只保留为尚未证实的探索性改进，不把它称为已验证的新方法。

这不是“方法完全无效”：在 61240 上，两种定向方法都通过合法的 Full 保持条件，使 FastV 自然回答从正确人名变成另一位人物，并在固定终点保持错误。它说明 complete-target 在这个样本上可行；它没有证明 complete-front 有可重复的额外收益。

## 样本和审核

六个样本与顺序在看题目、答案、图像和模型输出前冻结，见 [sample_selection_freeze.json](sample_selection_freeze.json)。没有更换任何样本，也没有运行那 20 个保留测试样本。需要说明：先前一次清单预览意外显示过保留 ID 452 的问答，因此 452 已标记为暴露，绝不称为 untouched；它没有被攻击或评测。

六题的 clean admission 为 2/6：

| ID | clean FastV 与数据集答案 | 状态 |
|---|---|---|
| 19153 | 传真号码不一致，ANLS 0.60 | `CLEAN_GATE_FAILED` |
| 46511 | `$12.26` 对 `$69.48`，ANLS 0 | `CLEAN_GATE_FAILED` |
| 4742 | `Morehead Ky`，ANLS 1.00 | admitted；冻结错误目标 `Marshall, KY` |
| 61240 | `Ruth Shiffrin`，ANLS 0.765 | admitted；冻结错误目标 `Mary E. Fordman` |
| 49550 | 输出 B，答案是 Control [C]，ANLS 0 | `CLEAN_GATE_FAILED` |
| 4464 | 输出 Other，答案是 Product Quality/Liability Issues，ANLS 0 | `CLEAN_GATE_FAILED` |

四道 clean 失败题保留在六题分母里，没有被攻击，也没有用新题替换。目标只对两道 clean 合格题生成：每题最多 24 个原始分支、最多 6 个完整续写；人工核对类型和源文后冻结。两种目标方法使用相同的错误答案与正确表达列表。

## 最终 PNG 结果

每条攻击都执行固定步数；拒绝更新也会消耗预算。最终图按预注册规则取最后一个被接受的 PNG，没有挑中途最好看的图。三组每题预算分别是 Native-FDP 40 次反向计算、complete-target 39 次、complete-front 40 次。六条轨迹均跑满，所有尝试都被接受；每一步 Full 文本和 token exact guard、FastV/Full 进程内重复检查通过。

| 样本 | 方法 | 最终 FastV 输出 | ANLS | 命中冻结目标 | 内容审核 | Full exact / PNG 独立复评 |
|---|---|---|---:|---|---|---|
| 4742 | Native-FDP | `Marehead Ky` | 0.909 | 不适用 | 拼写差异有歧义，不计内容失败 | 通过 |
| 4742 | complete-target | `Marehead Ky` | 0.909 | 否 | 拼写差异有歧义，不计内容失败 | 通过 |
| 4742 | complete-front | `Marehead Ky` | 0.909 | 否 | 拼写差异有歧义，不计内容失败 | 通过 |
| 61240 | Native-FDP | `Ruth Shiffrin, M.D.` | 0.588 | 否 | 正确人物＋后缀，答案保留 | 通过 |
| 61240 | complete-target | `Mary E. Fordman` | 0.000 | 是 | **`CONTENT_FAILURE`** | 通过 |
| 61240 | complete-front | `Mary E. Fordman` | 0.000 | 是 | **`CONTENT_FAILURE`** | 通过 |

4742 的证据页写明 `Morehead, KY`，而三个最终输出都是 `Marehead Ky`。这只是与正确地名差一个字母；没有足够证据认定为另一个实体，所以保守标记为有歧义、不计攻击成功。冻结目标 `Marshall, KY` 没有被任何组自然输出。

61240 的源文明确写着 Ruth Shiffrin。`Mary E. Fordman` 是不同人物、答案类型仍然正确，故是内容错误而不是大小写或格式变化。Native-FDP 在第 16、17 步一度输出其他错误人名，但第 18 步回到 Ruth Shiffrin，最终又输出 `Ruth Shiffrin, M.D.`。这些中间错误不算固定终点成功。

在固定六题分母下，最终内容失败数为 Native-FDP **0/6**、complete-target **1/6**、complete-front **1/6**；四个 clean 失败题作为攻击前无效样本留在分母。只看两个 clean 合格样本时则分别是 0/2、1/2、1/2。样本太少，不应把这些比例解释为总体成功率。

## complete-front 有没有额外价值？

61240 的逐步自然生成记录显示：

| 方法 | 首次自然命中错误目标 | 反向计算 | 生成调用 | FastV 评分前向 | Full 评分前向 | 该步累计墙钟时间 |
|---|---:|---:|---:|---:|---:|---:|
| complete-target | 第 9 步 | 27 | 45 | 18 | 9 | 174.2 秒 |
| complete-front | 第 6 步 | 24 | 30 | 18 | 6 | 130.3 秒 |

这是一条值得保留的**单样本成本信号**：complete-front 用更少的完整生成检查和较少的反向计算，更早达到同一错误答案。可是：

- 它没有在 complete-target 最终失败的样本上额外救活一个样本；两种方法都在 61240 成功、都在 4742 失败。
- 只有 1 个成功目标样本，无法证明“更早”会反复出现。
- Native-FDP 在 61240 也曾产生短暂的错误答案，说明“中途发生一次错误”并不是 complete-front 独有的现象；真正可见的差异是它更早达到预先指定的错误人名并维持到终点。
- 固定全程预算下 complete-front 用 40 次反向计算，complete-target 用 39 次；目前看到的是首次命中更快，不是终点质量更高。

因此目前最稳妥的判断是：**complete-front 可能提高达到指定错误答案的速度，但额外收益尚未重复验证；complete-target 已足以产生本轮唯一的终点内容失败。** 不应把它写成整体攻击成功率已经优于 Native-FDP，也不应据此声称 complete-front 已经胜过 complete-target。

## 复现与完整性口径

- 攻击过程的 `full_replay` / `fast_replay` 是同一攻击进程内重复检查，不叫独立进程复评。
- 最终六张 PNG 由新的 Python 进程、重新加载模型后从磁盘读取并复评。独立 evaluator 在各自新进程内又对同一已读图像重复推理两次，六条均一致。报告明确区分这两类检查。
- 六张最终图的 Full 文本和 token 都与各自 clean Full 完全相同；独立进程复评通过，K 均等于预期值 3699。
- 所有最终图的 uint8 L∞ 为 8；各 PNG SHA、输出文本、ANLS、keep-set hash 见 `independent_eval/`。
- 固定样本、目标、协议、程序及模型文件 SHA-256 见 [freeze_execution_lock.json](freeze_execution_lock.json)。冻结后没有修改攻击源代码。
- Smoke 阶段第一次启动有一次在目标记录加载处失败，以及一次 evaluator 相对路径错误；两者都没有形成可计分攻击结果。修正后两题三方法 smoke 通过，complete-front 在 4742 的新进程重复运行中梯度、扰动和 PNG 哈希一致。正式阶段使用修正后的路径并通过所有检查。

完整逐题判定和中间恢复事件在 [semantic_adjudication.json](semantic_adjudication.json)，预注册规则在 [protocol.md](protocol.md)，原始 clean 输出、轨迹和独立复评 JSON 均保留在本目录。

## 决策

**本轮不扩大样本、不打开保留测试集。** 按预注册的 `complete-front` 决策门槛，当前证据还没有达到“重复降低同一结果成本”的要求。主比较先报告为探索性开发结果：complete-target 与 complete-front 的最终效果相同，complete-front 在 61240 上有一次较早命中的信号。

若未来还要验证分叉模块，应另行冻结一个独立开发小队列与固定停止规则，专门重复“首次指定目标命中成本”比较；在重复结果出现前，方法主张采用较简单的 complete-target，complete-front 只作为待验证变体。20 道保留题继续封存。
