# Holdout provenance

冻结发生在任何 holdout 攻击之前。抽样没有看攻击结果，没有按 M 补齐，也没有按历史攻击难度筛选。

- 来源：`clean_baseline.csv` 与 `draw_questions(400, seed=42)` 的交集。
- 排除开发集：33923, 4912, 49242, 50743, 5169, 56776。
- 条件：clean Full ANLS ≥ 0.5，clean FastV r=0.25 ANLS ≥ 0.5，M > 1。
- 合格样本 100，种子 20261008，抽 20 题。
- manifest sha256（不含本字段）：8d9737ac302078ca9f5bb5e87dc19131d48dfd1913e45572bf3b97d9bf601d44。

抽中的 ID：

- 51484 M=4 Full 1.000 FastV 0.739
- 255 M=2 Full 1.000 FastV 0.667
- 57250 M=4 Full 1.000 FastV 1.000
- 50516 M=4 Full 1.000 FastV 0.667
- 46631 M=4 Full 1.000 FastV 1.000
- 50873 M=3 Full 1.000 FastV 1.000
- 37246 M=2 Full 1.000 FastV 1.000
- 64999 M=3 Full 1.000 FastV 1.000
- 4932 M=4 Full 0.667 FastV 0.667
- 51717 M=4 Full 1.000 FastV 0.971
- 63118 M=4 Full 1.000 FastV 0.667
- 63816 M=4 Full 0.600 FastV 0.667
- 56850 M=4 Full 0.556 FastV 0.556
- 43222 M=2 Full 1.000 FastV 1.000
- 59628 M=4 Full 0.842 FastV 0.842
- 6713 M=2 Full 1.000 FastV 1.000
- 57418 M=4 Full 1.000 FastV 1.000
- 1695 M=3 Full 1.000 FastV 1.000
- 56833 M=4 Full 1.000 FastV 1.000
- 56034 M=4 Full 0.625 FastV 0.625
