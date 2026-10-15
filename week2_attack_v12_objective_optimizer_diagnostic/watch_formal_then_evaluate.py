"""Wait for formal development trajectories, then run the independent evaluator."""
from __future__ import annotations
import json, subprocess, time
from pathlib import Path

ROOT=Path('/home/yuesiting/autodl-tmp/multimodal_attack_project')
HERE=ROOT/'week2_attack_v12_objective_optimizer_diagnostic'
PYTHON='/home/yuesiting/autodl-tmp/conda/envs/vattack/bin/python'
TAG='full_dev_frozen_v2'
REPORT=HERE/f'{TAG}_report.json'
LOG=HERE/f'{TAG}_post_eval.log'

def main():
    while True:
        if REPORT.exists():
            x=json.loads(REPORT.read_text())
            if x.get('status') in ('PASS_PROCESS_LEVEL','FAILED'):
                break
        time.sleep(30)
    if x.get('status') != 'PASS_PROCESS_LEVEL':
        (HERE/f'{TAG}_post_eval_status.json').write_text(json.dumps({'status':'NOT_RUN_FORMAL_FAILED','formal':x},indent=2)+'\n')
        return
    with LOG.open('w') as fh:
        rc=subprocess.call([PYTHON,str(HERE/'run_evaluator_matrix.py'),'--tag',TAG,'--all-dev'],cwd=ROOT,stdout=fh,stderr=subprocess.STDOUT)
    ev=HERE/f'evaluator_{TAG}_report.json'
    (HERE/f'{TAG}_post_eval_status.json').write_text(json.dumps({'status':'DONE' if rc==0 else 'FAILED','evaluator_report':str(ev)},indent=2)+'\n')
if __name__=='__main__': main()
