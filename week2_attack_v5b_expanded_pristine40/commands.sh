#!/usr/bin/env bash
# Commands actually used for v5b formal validation. Do not rerun the attack.
set -euo pipefail
cd /home/yuesiting/autodl-tmp/multimodal_attack_project
export PYTHONNOUSERSITE=1 PYTHONPATH=. PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python

$PY week2_attack_v5b_expanded_pristine40/verify_frozen_inputs.py
$PY week2_attack_v5b_expanded_pristine40/tests/test_cpu.py
CUDA_VISIBLE_DEVICES=6 $PY week2_attack_v5b_expanded_pristine40/run_formal_attack.py --dev50743 --steps 2 --tag smoke2
CUDA_VISIBLE_DEVICES=4 $PY week2_attack_v5b_expanded_pristine40/run_formal_attack.py --dev50743 --steps 25 --tag smoke25
CUDA_VISIBLE_DEVICES=6 $PY week2_attack_v5b_expanded_pristine40/run_random_noise.py --dev50743 --tag smoke25
CUDA_VISIBLE_DEVICES=6 $PY week2_attack_v5b_expanded_pristine40/replay_artifacts.py --dev50743 --tag smoke25
$PY week2_attack_v5b_expanded_pristine40/write_execution_lock.py

# Formal attack, started 11:15, all exit 0.
# CUDA_VISIBLE_DEVICES=0 $PY week2_attack_v5b_expanded_pristine40/run_formal_attack.py --shard 0 --nshards 4 --steps 200 --tag formal
# CUDA_VISIBLE_DEVICES=1 $PY week2_attack_v5b_expanded_pristine40/run_formal_attack.py --shard 1 --nshards 4 --steps 200 --tag formal
# CUDA_VISIBLE_DEVICES=4 $PY week2_attack_v5b_expanded_pristine40/run_formal_attack.py --shard 2 --nshards 4 --steps 200 --tag formal
# CUDA_VISIBLE_DEVICES=6 $PY week2_attack_v5b_expanded_pristine40/run_formal_attack.py --shard 3 --nshards 4 --steps 200 --tag formal

# Noise and replay were sharded the same way after each attack shard finished.
$PY week2_attack_v5b_expanded_pristine40/analyze_v5b.py
