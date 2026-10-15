# 47450 上下文杠杆因果诊断

状态：`NO_GO_CONTEXT_LEVERAGE_LOCAL_CAUSAL_EFFECT`

阶段 2 没有启动。v20 的 23 个源文件、配置、semantic manifest 和 47450 C/D 结果在实验后的 SHA256 与实验前一致。

## 分组

运行时 clean Full 和 clean FastV 都是 `Dr. or Mrs. William J. Darby`，token 和 keep-set 与 v20 一致。S_all 是 v20 方法 C 的 10 个 noEOS 位置。

- S_context，位置 0–4：`Dr` `.` ` or` ` Mrs` `.`，拼起来是 `Dr. or Mrs.`
- S_core，位置 5–9：` William` ` J` `.` ` Dar` `by`，与 v20 D 相同
- 二者不相交，并集等于 S_all

随机对照由种子 1701、1702、1703 抽出，大小与目标相同，且不等于目标集合：

| 名称 | 位置 | pieces |
|---|---|---|
| R_core_1701 / R_context_1701 | 3,4,6,7,9 | `Mrs` `.` `J` `.` `by` |
| R_core_1702 / R_context_1702 | 1,3,4,6,9 | `.` `Mrs` `.` `J` `by` |
| R_core_1703 | 0,1,2,3,9 | `Dr` `.` `or` `Mrs` `by` |
| R_context_1703 | 5,6,7,8,9 | 与 S_core 相同，但与 S_context 的 Jaccard 为 0 |

没有 activation self-patch。记录为 `SELF_PATCH_NOT_APPLICABLE_PIXEL_OBJECTIVE`，并保留了 NOOP 和 SHAM。

## 阶段 1

干净起点上，S_core 的 hard-margin 均值是 6.55。每个组从 delta=0 走一步，步长是 v20 step alpha 的 0.25、0.5、1 和 2 倍，然后投影、保存 PNG、从磁盘重读。Full exact 在全部局部步上都成立。自然文本始终是原姓名。

CONTEXT 在自然 keep-set 上的 S_core margin：

| 倍数 | margin | 相对 6.55 |
|---:|---:|---|
| 0.25 | 6.55 | 无 uint8 变化 |
| 0.50 | 6.85 | 升高 |
| 1.00 | 6.925 | 升高 |
| 2.00 | 6.8625 | 升高 |

冻结 clean keep-set 上的对应值是 6.55、6.7625、6.8、6.8125，方向相同。因此这不是“只在冻结 keep-set 下看起来下降”。CORE 和 ALL 的非零步也没有把 margin 降到 6.55 以下。NOOP 和 SHAM 保持 6.55。

没有一对相邻步长出现 CONTEXT 对 S_core margin 的下降，所以不再比较它是否强于随机对照，也不进入多步优化。v20 里 ALL 经过多步后曾经得到 `THE NUTRITION FOUNDATION, INC.`，不能用来补这个一步因果检验。

## 预注册问题

1. S_core、S_context 和上表中的随机集合就是本轮冻结结果。
2. CONTEXT 梯度方向没有降低 S_core margin。非零步长上 margin 升高。
3. 这个方向在投影和 PNG 回读之后仍然是升高，自然 keep-set 和冻结 keep-set 一致。
4. 阶段 2 的 CORE_ONLY、CONTEXT_ONLY、ALL、随机组和 NOOP 都是 `NOT_RUN`。
5. 没有形成可比较的上下文优势，因此不能说 CONTEXT 优于随机对照。
6. 没有独立上下文作用，也没有在本轮里重做 core-context interaction。
7. 本轮没有 oracle 或短暂内容失败；局部生成文本没有离开原姓名。
8. 局部步的 Full text/token exact 和 replay 都通过，非零步的 uint8 L∞ 为 1 或 2。
9. 阶段 2 的重复运行没有执行。阶段 0 的 clean token 与 v20 一致。
10. 没有资格扩展到第二个 development 样本，也不打开 unseen。
