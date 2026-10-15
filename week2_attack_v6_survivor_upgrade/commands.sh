#!/bin/bash
# Commands actually used for v6. v5b files are not written by these commands.
set -euo pipefail
ROOT=/home/yuesiting/autodl-tmp/multimodal_attack_project
PY=/home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python
cd "$ROOT"

# Stage 0 hashes were computed in the interpreter and stored in raw_hashes_before.json.
# Parent protocol SHA-256: 2e9a05f5d1be778de6321f783a99c790d85f75eae37e17e1370805fb66de0c87

# Stage I selector, CPU only.
PYTHONNOUSERSITE=1 "$PY" week2_attack_v6_survivor_upgrade/variant_a_audit/select_pixel.py

# Stage I replay. Started 2026-10-08T17:42+08, four idle GPUs, all EXIT 0.
# shard 0 GPU 0 pid 230913 log variant_a_audit/replay_s0.log
# shard 1 GPU 1 pid 230931 log variant_a_audit/replay_s1.log
# shard 2 GPU 4 pid 230934 log variant_a_audit/replay_s2.log
# shard 3 GPU 6 pid 230953 log variant_a_audit/replay_s3.log
for pair in 0:0 1:1 2:4 3:6; do
  shard=${pair%%:*}; gpu=${pair##*:}
  CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONPATH=. \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" week2_attack_v6_survivor_upgrade/variant_a_audit/replay_selected.py \
    --shard "$shard" --nshards 4 \
    > week2_attack_v6_survivor_upgrade/variant_a_audit/replay_s${shard}.log 2>&1 &
done

# Stage I 2x2 on the 12 selected PNGs. Started 2026-10-08T17:45:00+08.
# shard 0 GPU 0 pid 235466
# shard 1 GPU 1 pid 235468
# shard 2 GPU 4 pid 235470
# shard 3 GPU 6 pid 235472
# Each log ends with a mechanism JSON line and no traceback. JSON outputs are complete.
for pair in 0:0 1:1 2:4 3:6; do
  shard=${pair%%:*}; gpu=${pair##*:}
  CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONPATH=. \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" week2_attack_v6_survivor_upgrade/variant_a_audit/run_2x2_dev12.py \
    --shard "$shard" --nshards 4 \
    > week2_attack_v6_survivor_upgrade/variant_a_audit/logs/cells_s${shard}.log 2>&1 &
done

# Stage II health and calibration, one model on GPU 0. Both fit after score checkpointing.
# Health EXIT 0. Calibration EXIT 1: hinge gradient remained 0, so no coefficient was written.
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONPATH=. \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" week2_attack_v6_survivor_upgrade/run_bc.py --mode health --sample 1025 \
  --out week2_attack_v6_survivor_upgrade/logs/health_1025.json \
  > week2_attack_v6_survivor_upgrade/logs/health_1025.log 2>&1

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONPATH=. \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" week2_attack_v6_survivor_upgrade/run_bc.py --mode calibrate --sample 1025 \
  --out week2_attack_v6_survivor_upgrade/logs/calibration_1025.json \
  > week2_attack_v6_survivor_upgrade/logs/calibration_1025.log 2>&1

