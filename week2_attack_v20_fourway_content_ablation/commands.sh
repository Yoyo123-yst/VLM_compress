#!/usr/bin/env bash
# Commands actually used for v20. GPU policy: 0, 1, 4, 6 only.
set -euo pipefail
PY=/home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python
ROOT=/home/yuesiting/autodl-tmp/multimodal_attack_project
cd "$ROOT"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

$PY -m py_compile week2_attack_v20_fourway_content_ablation/*.py week2_attack_v20_fourway_content_ablation/tests/test_cpu.py
$PY -m unittest week2_attack_v20_fourway_content_ablation.tests.test_cpu
$PY week2_attack_v20_fourway_content_ablation/static_audit.py
$PY week2_attack_v20_fourway_content_ablation/freeze_inputs.py

# Admission, before any attack. Pass 2 is the frozen manifest.
CUDA_VISIBLE_DEVICES=0 $PY -u week2_attack_v20_fourway_content_ablation/run_admit.py

# Smoke: 60700 and 47450, four methods, four accepted updates.
CUDA_VISIBLE_DEVICES=0 $PY -u week2_attack_v20_fourway_content_ablation/run_train.py --method A --samples 60700,47450 --out week2_attack_v20_fourway_content_ablation/smoke --max-accepted 4 --max-attempts 8 --checkpoints 0,1,4
CUDA_VISIBLE_DEVICES=1 $PY -u week2_attack_v20_fourway_content_ablation/run_train.py --method B --samples 60700,47450 --out week2_attack_v20_fourway_content_ablation/smoke --max-accepted 4 --max-attempts 8 --checkpoints 0,1,4
CUDA_VISIBLE_DEVICES=4 $PY -u week2_attack_v20_fourway_content_ablation/run_train.py --method C --samples 60700,47450 --out week2_attack_v20_fourway_content_ablation/smoke --max-accepted 4 --max-attempts 8 --checkpoints 0,1,4
CUDA_VISIBLE_DEVICES=6 $PY -u week2_attack_v20_fourway_content_ablation/run_train.py --method D --samples 60700,47450 --out week2_attack_v20_fourway_content_ablation/smoke --max-accepted 4 --max-attempts 8 --checkpoints 0,1,4
CUDA_VISIBLE_DEVICES=0 $PY -u week2_attack_v20_fourway_content_ablation/run_train.py --method A --samples 60700 --out week2_attack_v20_fourway_content_ablation/smoke/repeat --max-accepted 4 --max-attempts 8 --checkpoints 0,1,4
CUDA_VISIBLE_DEVICES=1 $PY -u week2_attack_v20_fourway_content_ablation/evaluate_independent.py --run week2_attack_v20_fourway_content_ablation/smoke --method A --samples 60700 --out week2_attack_v20_fourway_content_ablation/smoke/independent_eval
$PY week2_attack_v20_fourway_content_ablation/smoke_check.py

# Formal tag full_dev_fourway_v1. Admitted order: 7006 39204 47450 60700.
CUDA_VISIBLE_DEVICES=0 $PY -u week2_attack_v20_fourway_content_ablation/run_train.py --method A --samples 7006,39204,47450,60700 --out week2_attack_v20_fourway_content_ablation/full_dev_fourway_v1 --max-accepted 25 --max-attempts 40 --checkpoints 0,1,5,10,15,20,25
CUDA_VISIBLE_DEVICES=1 $PY -u week2_attack_v20_fourway_content_ablation/run_train.py --method B --samples 7006,39204,47450,60700 --out week2_attack_v20_fourway_content_ablation/full_dev_fourway_v1 --max-accepted 25 --max-attempts 40 --checkpoints 0,1,5,10,15,20,25
CUDA_VISIBLE_DEVICES=4 $PY -u week2_attack_v20_fourway_content_ablation/run_train.py --method C --samples 7006,39204,47450,60700 --out week2_attack_v20_fourway_content_ablation/full_dev_fourway_v1 --max-accepted 25 --max-attempts 40 --checkpoints 0,1,5,10,15,20,25
CUDA_VISIBLE_DEVICES=6 $PY -u week2_attack_v20_fourway_content_ablation/run_train.py --method D --samples 7006,39204,47450,60700 --out week2_attack_v20_fourway_content_ablation/full_dev_fourway_v1 --max-accepted 25 --max-attempts 40 --checkpoints 0,1,5,10,15,20,25

for M in A B C D; do
  GPU=0; [ "$M" = B ] && GPU=1; [ "$M" = C ] && GPU=4; [ "$M" = D ] && GPU=6
  CUDA_VISIBLE_DEVICES=$GPU $PY -u week2_attack_v20_fourway_content_ablation/evaluate_independent.py --run week2_attack_v20_fourway_content_ablation/full_dev_fourway_v1 --method $M --samples 7006,39204,47450,60700 --out week2_attack_v20_fourway_content_ablation/independent_eval
done

CUDA_VISIBLE_DEVICES=0 $PY -u week2_attack_v20_fourway_content_ablation/run_noise.py --samples 7006 --out week2_attack_v20_fourway_content_ablation/noise_controls
CUDA_VISIBLE_DEVICES=1 $PY -u week2_attack_v20_fourway_content_ablation/run_noise.py --samples 39204 --out week2_attack_v20_fourway_content_ablation/noise_controls
CUDA_VISIBLE_DEVICES=4 $PY -u week2_attack_v20_fourway_content_ablation/run_noise.py --samples 47450 --out week2_attack_v20_fourway_content_ablation/noise_controls
CUDA_VISIBLE_DEVICES=6 $PY -u week2_attack_v20_fourway_content_ablation/run_noise.py --samples 60700 --out week2_attack_v20_fourway_content_ablation/noise_controls
$PY week2_attack_v20_fourway_content_ablation/summarize.py
