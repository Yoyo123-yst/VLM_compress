#!/usr/bin/env bash
# v22 safety-slack pilot. Old experiment trees are read-only.
set -euo pipefail
PY=/home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python
cd /home/yuesiting/autodl-tmp/multimodal_attack_project
mkdir -p week2_attack_v22_safety_slack_constraint_pilot/runtime week2_attack_v22_safety_slack_constraint_pilot/sealed_png
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
$PY -m py_compile week2_attack_v22_safety_slack_constraint_pilot/*.py week2_attack_v22_safety_slack_constraint_pilot/tests/test_cpu.py
$PY -m unittest week2_attack_v22_safety_slack_constraint_pilot.tests.test_cpu
$PY week2_attack_v22_safety_slack_constraint_pilot/provenance.py before
CUDA_VISIBLE_DEVICES=0 $PY -u week2_attack_v22_safety_slack_constraint_pilot/preflight.py \
  > week2_attack_v22_safety_slack_constraint_pilot/runtime/preflight.stdout \
  2> week2_attack_v22_safety_slack_constraint_pilot/runtime/preflight.stderr
CUDA_VISIBLE_DEVICES=0 $PY -u week2_attack_v22_safety_slack_constraint_pilot/train.py --trajectory T0 \
  > week2_attack_v22_safety_slack_constraint_pilot/runtime/T0.stdout \
  2> week2_attack_v22_safety_slack_constraint_pilot/runtime/T0.stderr &
p0=$!
CUDA_VISIBLE_DEVICES=1 $PY -u week2_attack_v22_safety_slack_constraint_pilot/train.py --trajectory T1 \
  > week2_attack_v22_safety_slack_constraint_pilot/runtime/T1.stdout \
  2> week2_attack_v22_safety_slack_constraint_pilot/runtime/T1.stderr &
p1=$!
CUDA_VISIBLE_DEVICES=4 $PY -u week2_attack_v22_safety_slack_constraint_pilot/train.py --trajectory T2 \
  > week2_attack_v22_safety_slack_constraint_pilot/runtime/T2.stdout \
  2> week2_attack_v22_safety_slack_constraint_pilot/runtime/T2.stderr &
p2=$!
fail=0
wait "$p0" || fail=1
wait "$p1" || fail=1
wait "$p2" || fail=1
if [[ "$fail" -ne 0 ]]; then
  echo "a trajectory exited nonzero" >&2
  exit 1
fi
CUDA_VISIBLE_DEVICES=0 $PY -u week2_attack_v22_safety_slack_constraint_pilot/independent_eval.py \
  > week2_attack_v22_safety_slack_constraint_pilot/runtime/independent.stdout \
  2> week2_attack_v22_safety_slack_constraint_pilot/runtime/independent.stderr
$PY week2_attack_v22_safety_slack_constraint_pilot/summarize.py
$PY week2_attack_v22_safety_slack_constraint_pilot/provenance.py after
