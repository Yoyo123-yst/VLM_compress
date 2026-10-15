#!/bin/bash
# Commands actually used. Working directory is the project root.
# Interpreter: /home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python
# Env: PYTHONUNBUFFERED=1 PYTHONPATH=. PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail
cd /home/yuesiting/autodl-tmp/multimodal_attack_project
PY=/home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python

# CPU tests. Passed before any new attack.
PYTHONNOUSERSITE=1 "$PY" week2_attack_v4_answer_aligned/tests/test_v4_cpu.py

# Stage A. GPU 4. Replay of the six v4 best images plus 2x2, margins, weighted gradients.
CUDA_VISIBLE_DEVICES=4 "$PY" -u week2_attack_v4_relaxed_full/run_stage_a.py

# Stage B smoke, then the six-question run. GPU 4 and 6, separate shard files.
CUDA_VISIBLE_DEVICES=4 "$PY" -u week2_attack_v4_relaxed_full/run_lambda.py --lambda-full 0.25 --steps 2 --ids 50743 --tag smoke2
CUDA_VISIBLE_DEVICES=4 "$PY" -u week2_attack_v4_relaxed_full/run_lambda.py --lambda-full 0.25 --steps 25 --ids 50743 --tag smoke25
CUDA_VISIBLE_DEVICES=4 "$PY" -u week2_attack_v4_relaxed_full/run_lambda.py --lambda-full 0.25 --steps 200 --tag lambda025 --shard 0 --nshards 2
CUDA_VISIBLE_DEVICES=6 "$PY" -u week2_attack_v4_relaxed_full/run_lambda.py --lambda-full 0.25 --steps 200 --tag lambda025 --shard 1 --nshards 2
PYTHONNOUSERSITE=1 PYTHONPATH=. "$PY" week2_attack_v4_relaxed_full/analyze_lambda.py lambda025

# Holdout was frozen before these attacks.
PYTHONNOUSERSITE=1 PYTHONPATH=. "$PY" week2_attack_v4_relaxed_full/freeze_holdout.py
MAN=week2_attack_v4_relaxed_full/holdout/frozen_manifest.json
CUDA_VISIBLE_DEVICES=4 "$PY" -u week2_attack_v4_relaxed_full/run_lambda.py --lambda-full 0.25 --steps 200 --tag holdout_lambda025 --manifest "$MAN" --shard 0 --nshards 2
CUDA_VISIBLE_DEVICES=6 "$PY" -u week2_attack_v4_relaxed_full/run_lambda.py --lambda-full 0.25 --steps 200 --tag holdout_lambda025 --manifest "$MAN" --shard 1 --nshards 2
CUDA_VISIBLE_DEVICES=0 "$PY" -u week2_attack_v4_relaxed_full/run_lambda.py --lambda-full 1.0 --steps 200 --tag holdout_lambda1 --manifest "$MAN" --shard 0 --nshards 2
CUDA_VISIBLE_DEVICES=1 "$PY" -u week2_attack_v4_relaxed_full/run_lambda.py --lambda-full 1.0 --steps 200 --tag holdout_lambda1 --manifest "$MAN" --shard 1 --nshards 2
PYTHONNOUSERSITE=1 PYTHONPATH=. "$PY" week2_attack_v4_relaxed_full/analyze_holdout.py
