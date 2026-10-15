# Commands actually executed.

cd /home/yuesiting/autodl-tmp/multimodal_attack_project
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python -m week2_attack_v7a_fdp_alignment_audit.audit_paths
