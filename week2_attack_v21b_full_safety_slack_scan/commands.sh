#!/usr/bin/env bash
# Frozen Full safety-slack scan. v20, v21, and v21a stay read-only.
set -euo pipefail
PY=/home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python
cd /home/yuesiting/autodl-tmp/multimodal_attack_project
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
$PY -m py_compile week2_attack_v21b_full_safety_slack_scan/*.py week2_attack_v21b_full_safety_slack_scan/tests/test_cpu.py
$PY -m unittest week2_attack_v21b_full_safety_slack_scan.tests.test_cpu
$PY -u week2_attack_v21b_full_safety_slack_scan/scan.py \
  > week2_attack_v21b_full_safety_slack_scan/runtime/scan.stdout \
  2> week2_attack_v21b_full_safety_slack_scan/runtime/scan.stderr
$PY -u week2_attack_v21b_full_safety_slack_scan/independent_eval.py \
  > week2_attack_v21b_full_safety_slack_scan/runtime/independent.stdout \
  2> week2_attack_v21b_full_safety_slack_scan/runtime/independent.stderr
$PY week2_attack_v21b_full_safety_slack_scan/summarize.py
