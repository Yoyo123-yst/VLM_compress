#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yuesiting/autodl-tmp/multimodal_attack_project"
PY="/home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python"
RUN_ROOT="$ROOT/week2_attack_v27_audit/confirmation_v1"
TARGETS="$ROOT/week2_attack_v25_full_answer_competition_pilot/frozen_targets_v26_step3.json"
SCRIPT="$ROOT/week2_attack_v25_full_answer_competition_pilot/run_iterative_compare.py"
mkdir -p "$RUN_ROOT/logs"

run_one() {
  local gpu="$1" sample="$2" method="$3" accepted="$4"
  local out="$RUN_ROOT"
  local log="$RUN_ROOT/logs/${sample}_${method}.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$SCRIPT" \
    --sample "$sample" --method "$method" --device 0 \
    --out "$out" --max-accepted "$accepted" --max-attempts 20 \
    --targets "$TARGETS" >"$log" 2>&1
}

# These are confirmation runs, not an independent holdout: all four samples
# were present in earlier development artifacts.  Budgets match v27:
# Native-FDP=20 accepted updates, complete-target=13, complete-front=10.
run_one 0 47450 native_fdp 20 & p1=$!
run_one 1 47567 complete_target 13 & p2=$!
run_one 4 51133 complete_front 10 & p3=$!
run_one 6 62941 native_fdp 20 & p4=$!
wait "$p1" "$p2" "$p3" "$p4"

run_one 0 47450 complete_target 13 & p1=$!
run_one 1 47567 complete_front 10 & p2=$!
run_one 4 51133 native_fdp 20 & p3=$!
run_one 6 62941 complete_target 13 & p4=$!
wait "$p1" "$p2" "$p3" "$p4"

run_one 0 47450 complete_front 10 & p1=$!
run_one 1 47567 native_fdp 20 & p2=$!
run_one 4 51133 complete_target 13 & p3=$!
run_one 6 62941 complete_front 10 & p4=$!
wait "$p1" "$p2" "$p3" "$p4"

echo "CONFIRMATION_V1_DONE"
