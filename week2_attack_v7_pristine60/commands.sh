# week2_attack_v7_pristine60 command log
# Python: /home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python
# Env: PYTHONNOUSERSITE=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2026-10-09T10:36+08 universe audit and 400-screen manifest
# python -m week2_attack_v7_pristine60.audit_universe
# decision UNIVERSE_AUDIT_PASS exclusion=320 unseen_multipage=3234

# 2026-10-09T10:41:35+08 clean screen, 100 each, EXIT wrapper 0
# shard 0 GPU 0 PID 885858
# shard 1 GPU 1 PID 885859
# shard 2 GPU 4 PID 885860
# shard 3 GPU 6 PID 885861
# eligible 143/400

# 2026-10-09 freeze_manifest then write_frozen
# python -m week2_attack_v7_pristine60.freeze_manifest
# python -m week2_attack_v7_pristine60.write_frozen
# manifest sha 62043dbdfb69c2de82fd9a149e8b260496c7f6571ec930bf7dcf6b9c4a86f327
# FROZEN sha 884aa0117847b2b24373a8d101a545664b048119f51b4148b9e870e1027c5e15

# 2026-10-09T10:49:00+08 health 1025 GPU 0
# runner exit 3 because keep-index list equality was false
# substantive hook/generate check passed inside evaluate_uint8; see logs/health_gate_note.json

# 2026-10-09T10:50:26+08 smoke 2 and 25 on 1025 GPU 0, both EXIT 0

# 2026-10-09T10:54:14+08 formal attack 15 each
# shard 0 GPU 0 PID 896734
# shard 1 GPU 1 PID 896735
# shard 2 GPU 4 PID 896736
# shard 3 GPU 6 PID 896737

# 2026-10-09T11:11:07+08:00 paused with SIGSTOP: 896734 GPU0, 896735 GPU1, 896736 GPU4, 896737 GPU6. Wrapper 896641 left in wait. Other users untouched.

# 2026-10-09T11:24:56+08:00 SIGKILL attack PIDs 896734 896735 896736 896737 to free GPUs 0,1,4,6. Other users not touched.

# 2026-10-09T11:41:05+08:00 resume GPU0 PID 923074 shard_s4 n=30; GPU6 PID 923075 shard_s5 n=29; kept completed 2288.

# 2026-10-09T21:51:19+08:00 attack finished shard_s4 30 shard_s5 29 shard_s3 1. Monitoring loop 944309 stopped.
