# Commands actually executed for the pristine60 closeout.
# Python: /home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python
# Env: PYTHONNOUSERSITE=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2026-10-09T22:49:49+08 repeat10 attack, then selector, then disk replay. EXIT 0 at 2026-10-10T00:53:04+08.
IDS=$(python3 -c "import json; print(','.join(json.load(open('week2_attack_v7_pristine60/repeat10/ids.json'))['ids']))")
for spec in "0:0" "1:6"; do
  shard=${spec%%:*}; gpu=${spec##*:}
  CUDA_VISIBLE_DEVICES=$gpu PYTHONNOUSERSITE=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python -m week2_attack_v7_pristine60.run_v7 \
    --mode attack --steps 200 --repeat --ids "$IDS" --shard $shard --nshards 2 \
    --out week2_attack_v7_pristine60/repeat10/attack/shard_s${shard}
done
PYTHONNOUSERSITE=1 PYTHONPATH=. /home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python -m week2_attack_v7_pristine60.select_run \
  --run week2_attack_v7_pristine60/repeat10/attack \
  --out week2_attack_v7_pristine60/repeat10/selector
for spec in "0:0" "1:6"; do
  shard=${spec%%:*}; gpu=${spec##*:}
  CUDA_VISIBLE_DEVICES=$gpu PYTHONNOUSERSITE=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python -m week2_attack_v7_pristine60.replay_record \
    --selected week2_attack_v7_pristine60/repeat10/selector/selected.csv \
    --shard $shard --nshards 2 \
    --out week2_attack_v7_pristine60/repeat10/replay/replay_s${shard}.json
done

# 2026-10-10T00:55:36+08 mechanism cells. EXIT 0 at 2026-10-10T00:59:40+08.
for spec in "0:0" "1:6"; do
  shard=${spec%%:*}; gpu=${spec##*:}
  CUDA_VISIBLE_DEVICES=$gpu PYTHONNOUSERSITE=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python -m week2_attack_v7_pristine60.run_2x2 \
    --selected week2_attack_v7_pristine60/attacks/selector/selected.csv \
    --shard $shard --nshards 2 \
    --out week2_attack_v7_pristine60/mechanism_audit/cells_s${shard}.json
done

# Closeout writer. The first run raised NameError and was not used. Later runs rewrote the reports.
PYTHONNOUSERSITE=1 PYTHONPATH=. /home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python -m week2_attack_v7_pristine60.finalize_closeout
