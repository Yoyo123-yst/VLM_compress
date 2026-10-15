# 原始记录完整性

状态：**PASS**

## lambda_1
- 样本数 20
- checkpoint steps: (0, 10, 25, 50, 75, 100, 150, 200)
- 重复 step: 无
- 缺字段: 0
- L∞ 违规: 0

## lambda_0_25
- 样本数 20
- checkpoint steps: (0, 10, 25, 50, 75, 100, 150, 200)
- 重复 step: 无
- 缺字段: 0
- L∞ 违规: 0

## K 与页数 M

M=2/3/4 对应不同合法 K，不能统一要求 3699。
- 推导：expected_k = official_keep_count(M * 3699, keep_ratio=RATIO); M=2->1850, M=3->2774, M=4->3699 at r=0.25
- K 审计状态：**PASS**

