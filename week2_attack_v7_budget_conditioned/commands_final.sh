# Commands actually executed.

cd /home/yuesiting/autodl-tmp/multimodal_attack_project
PYTHONNOUSERSITE=1 PYTHONPATH=. /home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python -m unittest week2_attack_v7_budget_conditioned.tests.test_fdp_cpu
PYTHONNOUSERSITE=1 PYTHONPATH=. /home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python -m week2_attack_v7_budget_conditioned.write_frozen

CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python -m week2_attack_v7_budget_conditioned.health_fdp

# Stage I, four shards, GPUs 0, 1, 4, 6. The first two launches exited on implementation errors before writing sample JSON. The third launch wrote 60/60.
for spec in "0:0" "1:1" "2:4" "3:6"; do
  shard=${spec%%:*}; gpu=${spec##*:}
  CUDA_VISIBLE_DEVICES=$gpu PYTHONNOUSERSITE=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python -m week2_attack_v7_budget_conditioned.stage1_diagnose \
    --shard $shard --nshards 4
done

PYTHONNOUSERSITE=1 PYTHONPATH=. /home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python -m week2_attack_v7_budget_conditioned.stage1_analyze
