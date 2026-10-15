#!/bin/bash
# 5-sample development post-hoc control. Does not train and does not rewrite sealed PNGs.
cd /home/yuesiting/autodl-tmp/multimodal_attack_project
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python \
  week2_attack_v9c_multiscale_rank_oblivious/target_eval_controls/evaluate_controls.py
