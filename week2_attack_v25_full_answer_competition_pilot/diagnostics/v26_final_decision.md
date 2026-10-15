# v26 最终开发决策

## 状态

`DEV_STEP3_LOCAL_WITNESS_STEP4_REQUIRED_NO_UNSEEN`

6 个预注册开发样本、3 种方法共 18 条轨迹均已完成。所有合法更新均通过 Full text/token exact、Full replay、FastV replay 和 uint8 L∞≤8；没有 OOM。Unseen 保持关闭。

## 当前证据

固定 selector 的完整错误目标命中为：

- Native-FDP：0/6；
- complete-target：0/6；
- complete-front：2/6（47567、50779）。

这比前一轮两样本 pilot 更有区分度，但仍不是稳定泛化证据。front 在 51133 第 9 步使目标首 token 出现过，但后续没有生成完整 `up to 30`；62941 的目标首 token 与 clean 共享月份前缀，变化发生在后续日期位置；6194 和 64506 连目标首 token 都没有出现。说明剩余失败集中在“目标序列后续位置/共享前缀”和“候选目标可达性”，不能再简单归因于 Full 权重。

## 是否继续

继续，但只做一个只读的目标可达性分析：利用已有 18 条轨迹，逐位置比较目标与自然输出的 token、完整目标分数、首分叉位置和 Full 最小 margin。禁止再改 loss、步长、权重或样本。若分析显示 front 的优势主要来自首分叉而后续位置普遍恢复，则下一步只允许做“后续目标位置约束”的单因素小实验；若首分叉本身也不稳定，则停止 complete-front 的方法化扩展，把它报告为局部强基线。

在该分析完成前，不启动新的 GPU 攻击、不打开 20 个 unseen 样本。

