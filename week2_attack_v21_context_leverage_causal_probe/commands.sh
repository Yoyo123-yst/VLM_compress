#!/usr/bin/env bash
# v21 causal probe. v20 is read-only. Phase 2 was not started.
set -euo pipefail
PY=/home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python
cd /home/yuesiting/autodl-tmp/multimodal_attack_project
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
$PY -m py_compile week2_attack_v21_context_leverage_causal_probe/*.py week2_attack_v21_context_leverage_causal_probe/tests/test_cpu.py
$PY -m unittest week2_attack_v21_context_leverage_causal_probe.tests.test_cpu
CUDA_VISIBLE_DEVICES=0 $PY -u week2_attack_v21_context_leverage_causal_probe/phase0.py
CUDA_VISIBLE_DEVICES=0 $PY -u week2_attack_v21_context_leverage_causal_probe/phase1.py
