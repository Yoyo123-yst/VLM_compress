#!/bin/bash
set -euo pipefail
cd /home/yuesiting/autodl-tmp/multimodal_attack_project
export PYTHONNOUSERSITE=1
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python

"$PY" -m unittest week2_attack_v7c_native60_readonly_remeasurement.tests.test_stats_cpu -v

CUDA_VISIBLE_DEVICES=0 "$PY" -m week2_attack_v7c_native60_readonly_remeasurement.run_shard --shard 0 &
CUDA_VISIBLE_DEVICES=1 "$PY" -m week2_attack_v7c_native60_readonly_remeasurement.run_shard --shard 1 &
CUDA_VISIBLE_DEVICES=4 "$PY" -m week2_attack_v7c_native60_readonly_remeasurement.run_shard --shard 2 &
CUDA_VISIBLE_DEVICES=6 "$PY" -m week2_attack_v7c_native60_readonly_remeasurement.run_shard --shard 3 &
wait

"$PY" -m week2_attack_v7c_native60_readonly_remeasurement.merge
