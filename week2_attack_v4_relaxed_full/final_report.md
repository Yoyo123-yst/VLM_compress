# λ=0.25 与独立 holdout（selector 审计后正式口径）

**Checkpoint 选择审计**（`selector_audit/`）后，holdout 主结果使用 **固定 step 200** 端点，不再把 oracle `best_strict` 当作普通攻击成功率。

ANLS 错误线保持 0.5。主比较 selector 为 `primary_fixed_step200`；`posthoc_min_compressed_margin` 与 `diagnostic_oracle_best_strict` 分层报告。

## 决定

1. 重放门禁通过，阶段 A 继续。
2. 开发集六题满足 B-GO，决定 `RELAXED_GO_TO_HOLDOUT`。没有跑阶段 C。
3. Holdout 配对完成后经 selector 审计，正式决定为 **`HOLDOUT_SUPPORTS_STRICT_COMP_ONLY_AT_FIXED_ENDPOINT`**（主 endpoint：λ=0.25 固定 step 200，strict 15/20）。

## 开发集，best_strict（诊断参考，非 holdout 主 endpoint）

| 样本 | λ=1 Full / FastV | λ=0.25 Full / FastV | λ=1 G | λ=0.25 G | λ=0.25 四格 |
| --- | --- | --- | ---: | ---: | --- |
| 50743 | 10-K / I | 10-K / 59 | 1.000 | 1.000 | FastV 错、Full 对 |
| 4912 | Joe Mitchell / Joe Mitchell | Joe Mitchell / H.R.J.KORMETTO, INC. | 0.000 | 1.000 | FastV 错、Full 对 |
| 56776 | Mar / May | Mar / May | 0.333 | 0.333 | 两边都对 |
| 49242 | 0.905 / 0.808 | 0.905 / 0.724 | 0.097 | 0.181 | 两边都对 |
| 33923 | 1.000 / 0.000 | 1.000 / 0.000 | 0.806 | 0.806 | FastV 错、Full 对 |
| 5169 | Mg / M | Mg / mg2p | 1.000 | 1.000 | FastV 错、Full 对 |

λ=1：strict CSFR 3/6，Full 错误 0/6，error-rate gap 3/6，mean G 0.539。

λ=0.25：strict CSFR 4/6，Full 错误 0/6，两边都错 0，反向失败 0，error-rate gap 4/6，mean D_F 0，mean D_C = mean G 0.720。六题 replay 和 ε 都通过，没有空输出。4912 是新增严格成功。原来的三个严格成功都还在。

## Holdout 主结果：固定 step 200

冻结样本 n=20，两种 λ 共用同一 manifest、初始化、200 步、ε=8/255 与同一模型。指标来自 **在线 checkpoint JSONL** 中 step=200 的真实 generate 记录；审计验证了日志完整性（`selector_audit/raw_integrity.json`、`raw_immutability_check.json`）。holdout 运行目录 **未** 为每题单独保存 step-200 的 `best.png`，因此不能把开发集六题 `best.png` 的 20/20 replay 直接套用到 fixed endpoint。

### λ=0.25（主 endpoint）

| 指标 | 值 |
| --- | ---: |
| strict compression-only（FastV 错且 Full 对） | **15/20** |
| strict CSFR | 0.75 |
| Wilson 95% strict CSFR | [0.531, 0.888] |
| Full 错误 | 0/20 |
| FastV 错误 | 15/20 |
| 两边都错 | 0/20 |
| 反向失败 | 0/20 |
| Full 解码文本与 step 0 逐字一致 | 20/20 |
| Full ANLS 与 step 0 一致 | 20/20 |
| mean G | 0.650 |
| INVALID_OUTPUT（所选 checkpoint） | 0 |

### λ=1（同主 endpoint，对照）

| 指标 | 值 |
| --- | ---: |
| strict compression-only | 12/20 |
| Full 错误 | 1/20（50873，step 200 两边都错） |
| mean G | 0.541 |

## Post-hoc min compressed margin

- λ=0.25 strict：**16/20**
- 选择时 **不使用** Full/FastV ANLS 或答案；规则为 `output_valid` 内最小 compressed margin，同 margin 取更早 step。
- 标签：**`POST_HOC_SELECTOR_AUDIT_NOT_PROSPECTIVE_EVIDENCE`**
- 当前数据是看到结果后的回溯；只有在新 holdout 运行前冻结该规则，才可称为 prospective selector。

## Oracle upper bound（`best_strict`）

- λ=0.25 strict：**17/20**
- 复现 `loss.select_best`：先 Full ANLS≥0.5，再最小化 FastV ANLS（tie-break：margin、linf、step）。
- 标签：**`GROUND_TRUTH_ORACLE_UPPER_BOUND`**
- Full error=0 受选择器条件影响 → **`FULL_SAFETY_IS_SELECTOR_CONDITIONED`**；不能当作无选择器条件下的部署成功率。

| 口径 | λ=0.25 strict | 选图时用 ANLS/答案 |
| --- | ---: | --- |
| 固定 step 200（主） | 15/20 | 否 |
| post-hoc min-margin | 16/20 | 否 |
| oracle best_strict | 17/20 | 是 |

## λ 比较（主 endpoint：fixed step 200）

|  | 值 |
| --- | ---: |
| λ=0.25 vs λ=1 strict | 15/20 vs 12/20 |
| candidate-only strict | 3 |
| baseline-only strict | 0 |
| exact McNemar p | 0.25 |
| mean ΔG | 0.128 |
| t-CI95（df=19） | [−0.023, 0.280] |
| percentile bootstrap CI95（seed=20261008, B=10000） | [0.0039, 0.280] |
| exact sign-flip p（6 个非零配对） | 0.125 |

在固定 200 步主 endpoint 上，mean ΔG 的 **t 区间包含 0**，而 **percentile bootstrap 区间不包含 0**；精确 McNemar 与精确 sign-flip 均未拒绝原假设。不同推断方法边界不一致，n=20 **不足以证明** λ=0.25 稳定优于 λ=1（**NOT_ESTABLISHED**）。

## 九个问题（更新 holdout 口径）

1. λ=0.25 在开发集与 holdout 主 endpoint 上均提高 FastV 严格失败率（开发 4/6；holdout 15/20 fixed）。相对 λ=1 的 **额外** 优势未确立。
2. 改善在 fixed endpoint 上仍为 strict compression-only（λ=0.25：both wrong 0）；oracle 17/20 不能混入主表。
3. Full 梯度压制仍是机制假说；λ=0.25 fixed endpoint 上 Full 文本 20/20 不变。
4. 六题机制诊断见 `diagnostics/diagnosis.md`（content / unexplained）。
5. EOS 仅部分样本主导 margin；见阶段 A token_margin。
6. 开发集与 holdout 主 endpoint 的 error-rate gap / mean G 均支持 compression amplification 描述，但 λ 优劣未确立。
7. Holdout 正式结论与开发集方向一致（strict comp-only 成立），但 **不能把 17/20 写成 holdout 主成功率**。
8. 证据支持 **严格 compression-only attack（固定 step 200 主 endpoint）**；不支持“λ=0.25 显著优于 λ=1”。
9. 下一步：冻结真正未见验证集；若用 min-margin，须事前预注册；不要继续无预注册扫 λ。

## 运行和异常

- CPU 测试通过。完整的 v4 `run_health.py` 没有重跑完：它会改写 v4 的 `FROZEN.json`。进程在写出文件后被停掉，原来的 `health.json`（PASS）还在。替代门禁是六题 best.png 的真实 generate 重放，6/6 一致。
- 2 步冒烟的攻击和重放是成功的，第一次写 CSV 时参数顺序写反，修完后重跑了 2 步和 25 步。25 步时 50743 已是可重放的严格成功，uint8 L∞ = 8/255。
- 正式 200 步没有 Traceback，没有跳过样本。开发集单题大约 1470–1503 秒。Holdout 单题大约 690–1504 秒，视页数而定。
- GPU 2、3、5、7 上是别人的 vLLM，没有动。开发集用 4 和 6。Holdout 同时用空闲的 0、1、4、6。
- 阶段 C 没有跑。今晚的新变体只有 λ=0.25。
- **Selector 审计**（CPU）：`selector_audit/recompute_selectors.py`；raw 文件哈希运行前后一致（`raw_immutability_check.json`）；K 按 M=2/3/4 分别校验（1850/2774/3699）。

## 文件

- `diagnostics/diagnosis.md`
- `results/lambda025/summary.md`、`decision.json`、`per_sample.csv`
- `results/holdout/paired.csv`（历史 oracle 口径；正式结论见 `selector_audit/`）
- `holdout/frozen_manifest.json`
- `selector_audit/`（主 endpoint 重算、正式 holdout 数字来源）
- `results/all_samples.csv`、`results/all_checkpoints.csv`
- `figs/lambda_comparison.png`、`figs/full_vs_fastv_degradation.png`、`figs/error_quadrants.png`
- `logs/`
