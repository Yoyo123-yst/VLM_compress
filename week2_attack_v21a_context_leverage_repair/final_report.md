# v21a：零扰动 replay 与一步梯度分离

状态：`CONTEXT_EFFECT_BLOCKED_BY_FULL_CONSTRAINT`

这只说明 47450 上，上下文 attack-only 方向有局部跨 token 作用，而当前 Native-FDP 联合更新把它抵消了。本轮没有多步训练，没有第二个样本，也没有打开 unseen。v20 和 v21 的 36 个冻结哈希在结束后仍然一致。

## 1–4. 零扰动 replay

五次全新进程都在 GPU 0 上串行运行。ORIGINAL、ZERO_PROJECTED、ZERO_DIRECT_SAVE 的 RGB raw hash 都是 `bbf94ce24477…`，uint8 L∞ 为 0，不同像素数为 0。三条路径的 processor tensor hash、K=3699、keep-set `4cdba19c…`、logits hash 和 margin hash 都相同，五次进程之间也相同。文件 SHA256 同样相同。阶段 A 是 `ZERO_REPLAY_EXACT`。

v21 里“零扰动 PNG 的 keep-set 不同”不是图像重建差异。同一张原始 RGB 图走 `processor.prepare` 得到 `4cdba19c…`，走 v21 的 `pixels_from_page` 拼接得到 `23d358e23fca…`。两条代码路径的预处理不同，像素没有不同。

## 5. CORE attack-only

干净 S_core hard-margin 均值是 6.6，soft loss 是 6.1142。CORE_ATTACK_ONLY 在 PNG 回读后：

| 步长 | hard mean | soft loss | 下降 token 数 | Full exact |
|---:|---:|---:|---:|---|
| 0.5 | 6.25 | 5.8203 | 4 | 通过 |
| 1.0 | 6.2 | 5.7656 | 4 | 通过 |
| 2.0 | 5.825 | 5.4865 | 4 | 通过 |

自然 keep-set 和冻结 keep-set 都下降。符号没有写反，soft loss 和 hard margin 同时下降，所以不是 surrogate/hard 代理错位。

## 6–8. CONTEXT attack-only 与 Full-FDP

CONTEXT_ATTACK_ONLY 的 S_core hard mean 是 6.45、6.4、6.3。冻结 keep-set 上是同样三个数。1.0 倍步长的逐 token 变化是：` William` −0.5，` J` −0.375，`.` −0.125，` Dar` +0.125，`by` −0.125。0.5 倍时 ` William` 和 ` J` 各降 0.5。这是多个 token 的平均下降，不是单个 0.125 的弱变化。自然输出仍然完整保留 `William J. Darby`。

`g_context` 与 `g_fdp` 的 cosine 是 −0.0675，像素符号冲突比例是 0.524。FDP_ONLY 把 hard mean 推到 6.9625、7.0375、6.725。CONTEXT_COMBINED 变成 6.8875、6.95、6.9375，方向从下降逆转成升高。Full exact、replay 和 L∞≤2 在这些步上都有记录并通过。

## 9. ALL 能否拆开

ALL_ATTACK_ONLY 的 hard mean 是 6.275、6.225、5.975，介于 CORE 和 CONTEXT 之间，且更接近 CORE。`g_core` 与 `g_all` 的 cosine 是 0.771，`g_context` 与 `g_all` 是 0.950；后者更高是因为 S_all 包含 S_context，不表示上下文单独主导了核心 margin。加入 FDP 后，CORE_COMBINED、CONTEXT_COMBINED、ALL_COMBINED 的均值都高于 6.6。一步联合更新没有留下可分离的上下文优势。

## 10. 终态

终态是 Full 约束冲突，不是 replay 阻塞，不是单 token 弱耦合，不是代理错位，也不是 strong no-go。attack-only 的局部作用在联合更新里被抵消，因此不能写成攻击已经有效，也不能据此继续多步 CONTEXT_ONLY。
