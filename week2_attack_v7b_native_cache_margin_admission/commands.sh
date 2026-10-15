#!/bin/bash
set -euo pipefail
cd /home/yuesiting/autodl-tmp/multimodal_attack_project
export PYTHONNOUSERSITE=1
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python

"$PY" -m unittest week2_attack_v7b_native_cache_margin_admission.tests.test_native_margin_cpu -v

CUDA_VISIBLE_DEVICES=0 "$PY" -m week2_attack_v7b_native_cache_margin_admission.tests.test_native_margin_gpu
