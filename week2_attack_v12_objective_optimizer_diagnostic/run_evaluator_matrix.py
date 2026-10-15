"""Run independent all-ratio sealed-PNG evaluators on four GPUs."""
from __future__ import annotations
import argparse, json, os, subprocess, time
from pathlib import Path

ROOT=Path('/home/yuesiting/autodl-tmp/multimodal_attack_project')
HERE=ROOT/'week2_attack_v12_objective_optimizer_diagnostic'
PYTHON='/home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python'
SMOKE_SAMPLES=('1025','60700','47450')
DEV_SAMPLES=('1025','2008','47450','60700','7006','39204')
GROUPS=('H0','H1','P0','P1','O0','O1'); GPUS=('0','1','4','6')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--tag',required=True); p.add_argument('--all-dev',action='store_true'); args=p.parse_args()
    samples = DEV_SAMPLES if args.all_dev else SMOKE_SAMPLES
    jobs=[(s,g) for s in samples for g in GROUPS]; pending=list(jobs); running={}
    logs=HERE/f'evaluator_{args.tag}_logs'; logs.mkdir(exist_ok=True)
    report={'status':'RUNNING','tag':args.tag,'all_dev':args.all_dev,'samples':samples,'jobs':{f'{s}/{g}':'PENDING' for s,g in jobs}}
    rp=HERE/f'evaluator_{args.tag}_report.json'; rp.write_text(json.dumps(report,indent=2)+'\n')
    while pending or running:
        for gpu in GPUS:
            if any(slot==gpu for *_,slot in running.values()) or not pending: continue
            s,g=pending.pop(0); key=f'{s}/{g}'; log=(logs/f'{s}_{g}_gpu{gpu}.log').open('w')
            env=os.environ.copy(); env.update({'CUDA_VISIBLE_DEVICES':gpu,'PYTHONNOUSERSITE':'1','PYTHONUNBUFFERED':'1','PYTHONPATH':str(ROOT),'TOKENIZERS_PARALLELISM':'false'})
            cmd=[PYTHON,str(HERE/'evaluate_sealed.py'),'--sample',s,'--group',g,'--tag',args.tag]
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            running[proc.pid]=(s,g,proc,log,gpu); report['jobs'][key]=f'RUNNING_GPU_{gpu}'
        for pid,(s,g,proc,log,gpu) in list(running.items()):
            rc=proc.poll()
            if rc is None: continue
            log.close(); report['jobs'][f'{s}/{g}']='DONE' if rc==0 else f'FAILED_RC_{rc}'; del running[pid]
        report['pending']=len(pending); report['running']=len(running); rp.write_text(json.dumps(report,indent=2)+'\n')
        if pending or running: time.sleep(2)
    bad=[k for k,v in report['jobs'].items() if v!='DONE']; report['status']='PASS_PROCESS_LEVEL' if not bad else 'FAILED'; report['failed']=bad
    rp.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2)); raise SystemExit(bool(bad))
if __name__=='__main__': main()
