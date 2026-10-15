# v26 Step 2 — 等计算预算比较

## 目的

检验 `complete_target` / `complete_front` 在 7006、60700 上的优势是否只是运行了更多更新，还是在相近反向传播预算下仍能达到 Native-FDP 达不到的完整错误答案。所有轨迹使用同一辅助页、扰动上限、Full exact guard、PNG 回读和独立 FastV replay 协议。

计算预算按每次接受更新的近似反向传播成本预注册：Native-FDP 20 次、complete-target 13 次、complete-front 10 次。实际墙钟时间另行报告，不能用接受步数替代。

## 结果

| 样本 | 方法 | 接受/尝试 | 墙钟(s) | 首次命中指定完整错误目标 | 最终 FastV 文本 | Full/PNG guard |
|---|---|---:|---:|---:|---|---|
| 7006 | Native-FDP | 20/20 | 316.6 | 未命中 | `Tobacco Company` | 全部通过 |
| 7006 | complete-target | 13/13 | 301.4 | 未命中 | `RJ Reynolds Tobacco Company` | 全部通过 |
| 7006 | complete-front | 10/10 | 256.4 | 第 8 个接受更新 | `Public Relations Dept.` | 全部通过 |
| 60700 | Native-FDP | 20/20 | 253.8 | 未命中 | `PPBS` | 全部通过 |
| 60700 | complete-target | 10/13 | 200.6 | 第 10 个接受更新 | `Dr. H. S. SORBS` | 全部通过 |
| 60700 | complete-front | 10/10 | 181.5 | 第 5 个接受更新 | `Dr. H. S. SORBS` | 全部通过 |

## 语义边界

7006 Native-FDP 的 `Tobacco Company` 是原答案 `RJ Reynolds Tobacco Company` 的内容保留/截短近邻，不计为指定完整错误目标命中；它可作为“发生表层或部分语义变化”的诊断状态。60700 Native-FDP 的 `Ppbs`、`Ppjs 14, 1967`、`PPBS #14, 1957` 也不计为指定目标命中，最终回到 `PPBS`。

## 判定

**Step 2 通过，但只支持“完整目标/前缀目标在当前两题上具有初步等预算优势”，不支持泛化或独立方法优越性。**

- 7006：complete-front 在较少接受更新和更短墙钟内命中；两个对照均未命中指定目标。该优势不是由更多接受更新解释的。
- 60700：complete-front 更早命中，complete-target 也命中；Native-FDP 未命中。该样本同样支持完整错误目标机制优于单纯压低当前答案。
- 目前只有 2 个样本，且目标是预注册冻结目标；不能打开 unseen，也不能把 front 的额外前缀计算称为普适优势。

## 下一步门槛

允许进入 Step 3：在攻击前冻结 4–6 个未参与本轮调参的 development 样本、候选生成规则、目标来源、计算预算、语义审核和 selector。不得根据攻击结果替换样本或目标。39204 继续记为 `NO_VALID_PREREGISTERED_COMPLETE_WRONG_TARGET`，不强行补目标。

若 Step 3 中 front 的优势只在少数样本出现、或目标覆盖率很低，则停止扩展并将其报告为受限的完整目标攻击基线；只有固定 selector 的 `CONTENT_FAILURE + Full exact` 在多题上稳定优于 Native-FDP，才考虑正式方法主张。

