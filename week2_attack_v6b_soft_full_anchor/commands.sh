#!/bin/bash
# v6b commands. Does not write v6 or v5b.
set -euo pipefail
ROOT=/home/yuesiting/autodl-tmp/multimodal_attack_project
PY=/home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python
cd "$ROOT"
PYTHONNOUSERSITE=1 PYTHONPATH=. "$PY" -m unittest week2_attack_v6b_soft_full_anchor.tests.test_cpu

# Health, 2026-10-08T18:23:40+08:00. One process per free GPU.
# 1025 GPU 0 pid 271416
# 6194 GPU 1 pid 271417
# 7006 GPU 4 pid 271418
# Each log ends with health_pass true. Selection gradient L2 is 0, so calibration blocked.
for spec in 1025:0 6194:1 7006:4; do
  sid=${spec%%:*}; gpu=${spec##*:}
  CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONPATH=. \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" week2_attack_v6b_soft_full_anchor/run_v6b.py --mode health --sample "$sid" \
    --out week2_attack_v6b_soft_full_anchor/logs/health_${sid}.json \
    > week2_attack_v6b_soft_full_anchor/logs/health_${sid}.log 2>&1 &
done
