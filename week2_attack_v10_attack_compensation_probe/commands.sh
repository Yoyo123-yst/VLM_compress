#!/bin/bash
# Development constructability probe. Does not modify v7d or v9 history.
set -euo pipefail
cd /home/yuesiting/autodl-tmp/multimodal_attack_project
mkdir -p week2_attack_v10_attack_compensation_probe/logs
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python
$PY week2_attack_v10_attack_compensation_probe/test_projection.py
$PY week2_attack_v10_attack_compensation_probe/run_probe.py init
CUDA_VISIBLE_DEVICES=0 $PY week2_attack_v10_attack_compensation_probe/baseline.py
CUDA_VISIBLE_DEVICES=0 $PY week2_attack_v10_attack_compensation_probe/run_probe.py smoke --sample 2008 --steps 2
CUDA_VISIBLE_DEVICES=0 $PY week2_attack_v10_attack_compensation_probe/run_probe.py smoke --sample 2008 --steps 25
mkdir -p week2_attack_v10_attack_compensation_probe/logs
# Full 100-step probe, one sample per free GPU. GPUs 5 and 7 are left untouched.
CUDA_VISIBLE_DEVICES=0 $PY -m week2_attack_v10_attack_compensation_probe.run_probe full --sample 1025 > week2_attack_v10_attack_compensation_probe/logs/1025.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY -m week2_attack_v10_attack_compensation_probe.run_probe full --sample 2008 > week2_attack_v10_attack_compensation_probe/logs/2008.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY -m week2_attack_v10_attack_compensation_probe.run_probe full --sample 47450 > week2_attack_v10_attack_compensation_probe/logs/47450.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY -m week2_attack_v10_attack_compensation_probe.run_probe full --sample 60700 > week2_attack_v10_attack_compensation_probe/logs/60700.txt 2>&1 &
CUDA_VISIBLE_DEVICES=4 $PY -m week2_attack_v10_attack_compensation_probe.run_probe full --sample 7006 > week2_attack_v10_attack_compensation_probe/logs/7006.txt 2>&1 &
CUDA_VISIBLE_DEVICES=6 $PY -m week2_attack_v10_attack_compensation_probe.run_probe full --sample 39204 > week2_attack_v10_attack_compensation_probe/logs/39204.txt 2>&1 &
wait
$PY week2_attack_v10_attack_compensation_probe/decide.py
