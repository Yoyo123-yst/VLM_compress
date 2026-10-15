#!/usr/bin/env bash
# v21a repair. v20 and v21 are read-only. One GPU, serial processes, no multi-step training.
set -euo pipefail
PY=/home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python
cd /home/yuesiting/autodl-tmp/multimodal_attack_project
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
$PY -m py_compile week2_attack_v21a_context_leverage_repair/*.py week2_attack_v21a_context_leverage_repair/tests/test_cpu.py
$PY -m unittest week2_attack_v21a_context_leverage_repair.tests.test_cpu
$PY week2_attack_v21a_context_leverage_repair/provenance.py before
for run in 1 2 3 4 5; do
  $PY -u week2_attack_v21a_context_leverage_repair/replay_worker.py --run "$run" \
    > "week2_attack_v21a_context_leverage_repair/runtime/replay_${run}.stdout" \
    2> "week2_attack_v21a_context_leverage_repair/runtime/replay_${run}.stderr"
done
$PY week2_attack_v21a_context_leverage_repair/compare_replays.py
$PY -u week2_attack_v21a_context_leverage_repair/direction_probe.py \
  > week2_attack_v21a_context_leverage_repair/runtime/direction.stdout \
  2> week2_attack_v21a_context_leverage_repair/runtime/direction.stderr
$PY week2_attack_v21a_context_leverage_repair/provenance.py after
