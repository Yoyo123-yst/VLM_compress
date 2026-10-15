# 建议替换 final_report 中 holdout 主结论的表述（勿直接覆盖原文件）

## 主结果（Primary endpoint）

Holdout n=20，**固定 step 200**，λ=0.25：strict compression-only **15/20**，Full 错误率 **0.000**，FastV 错误率 **0.750**，mean G **0.650**。Full 解码文本与 step 0 一致 **20/20**。

## 次结果（Post-hoc，非预注册）

最小 compressed margin（选择时不用 ANLS）：strict **16/20**。  
**POST_HOC_SELECTOR_AUDIT_NOT_PROSPECTIVE_EVIDENCE**

## 诊断上界（Oracle）

现有 best_strict：**17/20**。  
**GROUND_TRUTH_ORACLE_UPPER_BOUND** — 不能当作普通攻击成功率。

## λ 比较

主 endpoint 上 λ=0.25 strict 15/20 vs λ=1 12/20；McNemar p=0.25；sign-flip p=0.125（6 个非零配对）；mean ΔG=0.128，t-CI95 包含 0，bootstrap-CI95 不包含 0。  
λ=0.25 的绝对 strict compression-only 很强，但相对 λ=1 的额外优势 **NOT_ESTABLISHED**（不得写显著优于）。
