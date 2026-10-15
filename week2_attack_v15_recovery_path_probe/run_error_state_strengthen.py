"""Strengthen an already observed legal FastV error from a frozen PNG."""
import copy, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT=Path('/home/yuesiting/autodl-tmp/multimodal_attack_project')
HERE=ROOT/'week2_attack_v15_recovery_path_probe'
V13=ROOT/'week2_attack_v13_branch_reachability_probe'
sys.path[:0]=[str(ROOT),str(ROOT/'week1_mechanism'),str(ROOT/'gate_experiment'),str(ROOT/'week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation'),str(V13)]
import run_branch_probe as probe
from week2_attack_v11_objective_optimizer_diagnostic import run_v11 as base
from week2_attack_v7d_bplus_dev_protocol_and_paired_evaluation import run_v7d as v7d
from week2_attack_v4_answer_aligned.run_smoke import page_uint8, project
from week2_attack_v4_answer_aligned.common import keep_hash
from metrics_task import anls_score

SAMPLE='60700'; RATIO=.25; EPS=8/255; STEPS=10; LR=2/255; LAMBDA=probe.LAMBDA_FULL
START=V13/'runs/state_retention_v1/60700/AWARE_FREEZE/step_011.png'
REFERENCE=[4406,16680,151645]  # naturally observed PPSB at the frozen state
OUT=HERE/'construction'/'60700'/'ERROR_SEQ_STRENGTHEN'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
  probe.configure_determinism(); model,processor=base.load_model_frozen(); model.eval()
  item=v7d.load_dev(SAMPLE); start=Image.open(START).convert('RGB')
  attacked=list(item['images']); attacked[item['aux_idx']]=start
  inputs=v7d.prepare(processor,attacked,item['meta']['question']); spans=v7d.spans_of(inputs,int(model.config.image_token_index),len(attacked))
  state={'inputs':inputs,'spans':spans,'page':v7d.page_to_float(start).to('cuda'),'base_page':v7d.page_to_float(item['images'][item['aux_idx']]).to('cuda')}
  clean=v7d.load_dev(SAMPLE); clean_inputs=v7d.prepare(processor,clean['images'],clean['meta']['question'])
  clean_spans=v7d.spans_of(clean_inputs,int(model.config.image_token_index),len(clean['images']))
  full0,_=v7d.generate_tokens(model,clean_inputs,clean_inputs['pixel_values'],clean_spans,None)
  clean_text=v7d.decode(processor,torch.tensor(full0)).strip()
  state['clean_text']=clean_text; state['clean_tokens']=full0
  with torch.no_grad():
    cr=v7d.full_token_rows(model,clean_inputs,v7d.page_to_float(item['images'][item['aux_idx']]).to('cuda'),item['aux_idx'],full0,False)
    state['clean_hard']=torch.stack([v7d.hard_margin_row(r,t) for r,t in zip(cr,full0)]).detach()
  opt_delta=(state['page']-state['base_page']).detach().clone().requires_grad_(True); opt=torch.optim.Adam([opt_delta],lr=LR)
  OUT.mkdir(parents=True,exist_ok=True); records=[]
  for step in range(STEPS+1):
    page=(state['base_page']+opt_delta.detach()).clamp(0,1); img=page_uint8(state['base_page'],page-state['base_page']); p=OUT/f'step_{step:03d}.png'; img.save(p)
    full=base.full_exact_check(model,processor,item,state,img)
    attacked2=list(item['images']); attacked2[item['aux_idx']]=img; ii=v7d.prepare(processor,attacked2,item['meta']['question']); ss=v7d.spans_of(ii,int(model.config.image_token_index),len(attacked2)); ft,kept=v7d.generate_tokens(model,ii,ii['pixel_values'],ss,RATIO); txt=v7d.decode(processor,torch.tensor(ft)).strip(); an=anls_score(txt,list(item['meta']['answers']))
    records.append({'step':step,'fast_text':txt,'fast_tokens':ft,'fast_anls':an,'endpoint_failure':bool(an<.5),'full_exact':bool(full['full_text_exact'] and full['full_token_exact'] and full['full_replay']),'full':full,'keep_hash':keep_hash(kept),'linf_uint8':int(np.max(np.abs(np.asarray(img,dtype=np.int16)-np.asarray(item['images'][item['aux_idx']],dtype=np.int16)))),'png_sha256':sha(p)})
    print(json.dumps(records[-1],ensure_ascii=False),flush=True)
    if step==STEPS: break
    opt.zero_grad(set_to_none=True); leaf=state['page']+opt_delta
    rows,_,kept2=probe.fast_rows_for_reference(model,state,item,leaf,REFERENCE)
    # complete target sequence negative log-probability / margin surrogate
    attack=v7d.mean_soft(rows,REFERENCE); attack.backward(); g=opt_delta.grad.detach().clone(); opt_delta.grad=None
    fdp,_=base.fdp_value(model,state,item,leaf); (LAMBDA*fdp).backward(); opt_delta.grad.add_(g)
    old=opt_delta.detach().clone(); oldopt=copy.deepcopy(opt.state_dict()); opt.step(); project(opt_delta,state['base_page'],EPS); cand=page_uint8(state['base_page'],(state['base_page']+opt_delta.detach()).clamp(0,1)-state['base_page']); guard=base.full_exact_check(model,processor,item,state,cand)
    if not (guard['full_text_exact'] and guard['full_token_exact'] and guard['full_replay']):
      with torch.no_grad(): opt_delta.copy_(old)
      opt.load_state_dict(oldopt)
    del rows,leaf,fdp,attack,g,ii,ss
    torch.cuda.empty_cache()
  (OUT/'trajectory.json').write_text(json.dumps({'sample':SAMPLE,'start_png':str(START.relative_to(ROOT)),'reference':REFERENCE,'steps':records},indent=2,ensure_ascii=False)+'\n')
if __name__=='__main__': main()
