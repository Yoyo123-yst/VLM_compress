#!/usr/bin/env bash
# Pristine40 protocol — executed 2026-10-08. Attack blocked at pool stage.
set -euo pipefail
cd /home/yuesiting/autodl-tmp/multimodal_attack_project
export PYTHONNOUSERSITE=1 PYTHONPATH=.
PY=/home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python

$PY week2_attack_v5_pristine40/build_exclusions.py
$PY week2_attack_v5_pristine40/build_pool.py
# BLOCKED: pristine_pool_size=0 — no freeze, clean_gate, or GPU attack below.

# GPU policy: prefer 6, then 4; dual-shard use 6 + 4.
# Engineering smoke (dev 50743 only):
# CUDA_VISIBLE_DEVICES=6 $PY week2_attack_v5_pristine40/run_smoke.py --steps 2
# CUDA_VISIBLE_DEVICES=4 $PY week2_attack_v5_pristine40/run_smoke.py --steps 25
# Future 40-sample attack (only if pristine pool >= 40):
# CUDA_VISIBLE_DEVICES=6 $PY week2_attack_v5_pristine40/run_attack.py --shard 0 --nshards 2 &
# CUDA_VISIBLE_DEVICES=4 $PY week2_attack_v5_pristine40/run_attack.py --shard 1 --nshards 2 &
