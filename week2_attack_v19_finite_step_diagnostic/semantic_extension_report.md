# Content-objective development extension

这是对 v19 核心内容目标的少量开发集扩展，不是最终方法结果，也没有访问 unseen。

## Protocol

- 60700：重复 `error` 起点，并复查 `recovery` 起点；
- 2008、47450：从 canonical clean auxiliary page 开始；
- 每条轨迹使用 natural/frozen 两种选择、两个步长（0.5/255、1.0/255）、4 次更新；
- 每一步继续执行真实 Full exact/replay、PNG 回读和 uint8 L∞ 检查；
- 语义标签由预注册规则产生，不由 ANLS 单独决定：
  `ANSWER_PRESERVED`、`ANSWER_PRESERVED_AUGMENTED`、`CONTENT_FAILURE`。

## Final endpoint summary

| sample/state | final settings | strict ANLS<0.5 | semantic content failure | Full exact failures |
|---|---:|---:|---:|---:|
| 60700 error (original) | 4 | 2/4 natural, 3/4 frozen | `PPSB`/`Ppjs` are content failures; `PPBS` preserved | 0 |
| 60700 error (repeat) | 4 | 1/4 | 1 `PPSB`, 2 `Ppjs` | 0 |
| 60700 recovery | 4 | 3/4 | 3 `PPSB`, 1 `PPBS` preserved | 0 |
| 2008 clean | 4 | 2/4 | `February 23, 1968` is a real date-content error; `March 1956` is also incorrect | 0 |
| 47450 clean | 4 | 0/4 | all preserve `William J. Darby` (augmented formatting only) | 0 |
| 7006 near (existing) | 8 | 0/8 | `Tobacco Company` is treated as preserved augmented answer | 0 |

## Interpretation

`PPSB` is not merely a suffix or punctuation artifact: it changes the core
answer token sequence `PPBS` and is therefore a genuine content failure under
the registered rule. However, the repeated 60700 run did not reproduce the
same PNG/output for every `(mode, alpha)` pair; the phenomenon is recurrent but
not yet stable enough to call deterministic.

The extension therefore supports only this limited claim:

> The core-content objective can induce genuine content changes on 60700 and
> can produce a date-content error on 2008 while preserving Full exact output,
> but the effect is sample-dependent and not yet a stable general method.

47450 is a negative control for semantic preservation. 7006 remains a
near-answer/recovery case rather than a demonstrated content failure.

## Artifacts

- `results_content_60700_repeat.json`
- `results_content_60700_recovery_ext.json`
- `results_content_2008_clean.json`
- `results_content_47450_clean.json`
- `semantic_extension_report.json`
- `semantic_extension_report.py`
