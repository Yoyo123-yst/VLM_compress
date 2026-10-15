#!/usr/bin/env bash
set -euo pipefail
cd /home/yuesiting/autodl-tmp/multimodal_attack_project
export PYTHONNOUSERSITE=1 PYTHONPATH=. PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python
DIR=week2_attack_v11a_aux_scope_amendment_objective_pair
mkdir -p "$DIR/logs" "$DIR/per_sample"
"$PY" "$DIR/run_v11a.py" init
CUDA_VISIBLE_DEVICES=0 "$PY" "$DIR/run_v11a.py" phase_a
# H0 only after phase A prints PASS. H1 only after compare_h0 exits 0.
# GPUs 5 and 7 are not used.
