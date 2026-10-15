#!/usr/bin/env bash
set -euo pipefail
cd /home/yuesiting/autodl-tmp/multimodal_attack_project
export PYTHONNOUSERSITE=1 PYTHONPATH=. PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python
DIR=week2_attack_v11b_h0_update_forensics
mkdir -p "$DIR/logs"
"$PY" "$DIR/run_v11b.py" provenance
CUDA_VISIBLE_DEVICES=0 "$PY" "$DIR/run_v11b.py" drive
