#!/bin/bash
# Development multi-seed noise diagnostic. Does not train and does not edit sealed attack PNGs.
set -euo pipefail
cd /home/yuesiting/autodl-tmp/multimodal_attack_project
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 /home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python \
  week2_attack_v9c_multiscale_rank_oblivious/noise_multiseed_diagnostic/prepare_full_gate.py
CUDA_VISIBLE_DEVICES=0 /home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python \
  week2_attack_v9c_multiscale_rank_oblivious/noise_multiseed_diagnostic/evaluate_fastv_diagnostic.py
