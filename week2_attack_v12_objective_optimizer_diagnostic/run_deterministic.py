"""Deterministic isolated runner for the six objective/update diagnostic groups."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('VATTACK_ATTN_IMPLEMENTATION', 'sdpa')
os.environ.setdefault('VATTACK_DETERMINISTIC_RESIZE', '1')
os.environ.setdefault('VATTACK_RESIZE_MODE', 'bilinear')
ROOT=Path('/home/yuesiting/autodl-tmp/multimodal_attack_project')
HERE=ROOT/'week2_attack_v12_objective_optimizer_diagnostic'
sys.path[:0]=[str(ROOT),str(ROOT/'week1_mechanism'),str(ROOT/'gate_experiment')]
import torch
import week2_attack_v11_objective_optimizer_diagnostic.run_v11 as base
base.HERE=HERE
base.IDS=('1025','60700')

def configure():
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sample',required=True)
    p.add_argument('--group',required=True,choices=sorted(base.GROUPS))
    p.add_argument('--steps',type=int,default=3)
    p.add_argument('--tag',default='smoke')
    p.add_argument('--checkpoint-mode',choices=('dense','frozen'),default=None,
                   help='dense=every step (smoke); frozen=the protocol 8 checkpoints (formal)')
    args=p.parse_args()
    configure(); HERE.mkdir(parents=True,exist_ok=True)
    mode = args.checkpoint_mode or ('dense' if args.steps <= 3 else 'frozen')
    base.CHECKPOINTS = tuple(range(args.steps + 1)) if mode == 'dense' else (0, 10, 25, 50, 75, 100, 150, 200)
    model,processor=base.load_model_frozen(); base.install_counters(model)
    out=HERE/'runs'/args.tag/args.group/args.sample
    result=base.train_one(model,processor,args.sample,args.group,args.steps,out)
    (out/'determinism.json').write_text(json.dumps({'env':{'cublas':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'attn':os.environ.get('VATTACK_ATTN_IMPLEMENTATION'),'resize':os.environ.get('VATTACK_RESIZE_MODE'),'deterministic':torch.are_deterministic_algorithms_enabled()},'sample':args.sample,'group':args.group,'steps':args.steps,'checkpoint_mode':mode,'checkpoints':list(base.CHECKPOINTS),'status':'DONE'},indent=2)+'\n')
    print(json.dumps({'status':'DONE','sample':args.sample,'group':args.group,'steps':args.steps,'checkpoint_mode':mode,'oracle_any_joint_success':result['oracle_any_joint_success'],'full_exact_all':all(x['full_text_exact'] for x in result['curve_selector_view'])}),flush=True)
if __name__=='__main__': main()
