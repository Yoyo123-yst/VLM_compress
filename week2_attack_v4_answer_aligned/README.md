# v4：答案对齐的压缩专属攻击

只改第一张非证据页，ε=8/255，r=0.25。主目标是压缩 teacher-forcing 的未截断平均 margin，不是 aux share。Full 项是 KL，两项各自除以在 50743 上冻结的像素梯度 L2 范数后，λ=1。

δ=0 时 KL 的梯度是 0。若它的 L2 范数低于 1e-6，尺度改用同一样本上固定的 +1/255 平移，不按题调整。

成功只看真实 generate 的 ANLS。margin、KL 和 keep set 是诊断。evidence_idx 不进入损失。

开发集是 FROZEN.json 里的 6 题。先 CPU 测试，再健康检查，再 2 步、25 步，最后才跑 200 步。
